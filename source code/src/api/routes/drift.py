"""
GET  /api/drift/messages/{pseudonym}   -> that user's scored messages (drift/z-scores, no raw content stored)
GET  /api/drift/explain/{pseudonym}    -> watsonx.ai plain-language explanation of why they were flagged
"""
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.data.store import read_df
from src.features.drift_scoring import feature_contributions
from src.governance.access_control import get_current_role, check_permission
from src.governance.audit_log import log_event
from src.models.watsonx import client as watsonx_client
from src.models.watsonx.client import explain_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# Columns considered safe to expose via the API — explicitly excludes raw
# message "content" even though it's present in scored_messages in-process,
# reinforcing the "pseudonyms + derived features only" boundary at the API layer.
EXPOSED_COLUMNS = [
    "user", "date", "drift_score", "n_flagged", "flagged_features",
    "sentiment_polarity", "urgency_score", "readability_flesch",
    "anomaly_score", "is_anomaly",
    # Communication-network drift (src/dsa/communication_graph.py) — headers
    # only, never raw addresses.
    "contact_degree", "interaction_concentration",
]


def _load_scored_messages() -> pd.DataFrame:
    df = read_df("scored_messages")
    if df is None:
        raise HTTPException(status_code=503, detail="Pipeline has not been run yet. Run `python -m src.pipeline`.")
    return df


@router.get("/messages/{pseudonym}")
def get_user_messages(pseudonym: str, role: str = Depends(get_current_role)):
    check_permission(role, "view_drift")
    df = _load_scored_messages()
    user_df = df[df["user"] == pseudonym]
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No scored messages for this pseudonym.")
    cols = [c for c in EXPOSED_COLUMNS if c in user_df.columns]
    log_event("api_access", pseudonym, {"endpoint": "get_user_messages"}, actor=role)
    return user_df[cols].to_dict(orient="records")


@router.get("/explain/{pseudonym}")
def get_drift_explanation(pseudonym: str, role: str = Depends(get_current_role)):
    check_permission(role, "view_drift")

    user_risk = read_df("user_risk")
    if user_risk is None:
        raise HTTPException(status_code=503, detail="Pipeline has not been run yet.")
    row = user_risk[user_risk["user"] == pseudonym]
    if row.empty:
        raise HTTPException(status_code=404, detail="User pseudonym not found.")

    summary = row.iloc[0].to_dict()
    explanation = explain_drift(pseudonym, summary)

    log_event("watsonx_explanation_generated", pseudonym,
              {"explanation_generated": explanation is not None}, actor=role)

    if explanation is None:
        # explain_drift() -> _get_model() never raises (client.py catches
        # init/call errors internally and returns None), so we can safely
        # distinguish *why* it's None instead of collapsing both cases into
        # one misleading "not configured" message.
        if not watsonx_client.WATSONX_ENABLED:
            explanation = (
                "watsonx.ai not configured — set WATSONX_API_KEY/WATSONX_PROJECT_ID in .env."
            )
        elif watsonx_client._model_init_failed:
            explanation = (
                "watsonx.ai explanation unavailable — the watsonx client failed to "
                "initialize (check credentials, network access, and the ibm-watsonx-ai "
                "package). See server logs for details."
            )
        else:
            explanation = (
                "watsonx.ai explanation unavailable — the model call failed. "
                "See server logs for details."
            )

    return {
        "user_pseudonym": pseudonym,
        "summary": summary,
        "explanation": explanation,
    }


@router.get("/attribution/{pseudonym}")
def get_feature_attribution(pseudonym: str, role: str = Depends(get_current_role)):
    """
    Deterministic, auditable breakdown of *why* this user's drift score is
    what it is — each of the ~11 weighted z-score features and its exact
    share of the score shown in the table, computed with the same math
    src/features/drift_scoring.score_drift() uses (not an approximation,
    not an LLM call — this is the rule-based layer the watsonx.ai
    explanation in /explain sits on top of, and it stays correct even when
    watsonx.ai isn't configured).

    Uses the message with this user's single highest drift_score (their
    most-flagged instance) as the explained example, rather than averaging
    across their history, so the breakdown corresponds to one concrete,
    inspectable data point — the same framing as an instance-level SHAP
    explanation.
    """
    check_permission(role, "view_drift")
    df = _load_scored_messages()
    user_df = df[df["user"] == pseudonym]
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No scored messages for this pseudonym.")

    z_cols = [c for c in user_df.columns if c.startswith("z_")]
    if not z_cols:
        return {"user_pseudonym": pseudonym, "date": None, "drift_score": 0.0, "contributions": []}

    peak_row = user_df.loc[user_df["drift_score"].idxmax()]
    z_row = {c: peak_row[c] for c in z_cols}
    contributions = feature_contributions(z_row)

    log_event("api_access", pseudonym, {"endpoint": "get_feature_attribution"}, actor=role)
    return {
        "user_pseudonym": pseudonym,
        "date": str(peak_row.get("date", "")),
        "drift_score": round(float(peak_row["drift_score"]), 4),
        "contributions": contributions,
    }
