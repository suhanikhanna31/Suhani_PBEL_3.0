"""
GET  /api/drift/messages/{pseudonym}   -> that user's scored messages (drift/z-scores, no raw content stored)
GET  /api/drift/explain/{pseudonym}    -> watsonx.ai plain-language explanation of why they were flagged
"""
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.config import DATA_PROCESSED
from src.governance.access_control import get_current_role, check_permission
from src.governance.audit_log import log_event
from src.models.watsonx.client import explain_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# Columns considered safe to expose via the API — explicitly excludes raw
# message "content" even though it's present in scored_messages.csv in-process,
# reinforcing the "pseudonyms + derived features only" boundary at the API layer.
EXPOSED_COLUMNS = [
    "user", "date", "drift_score", "n_flagged", "flagged_features",
    "sentiment_polarity", "urgency_score", "readability_flesch",
]


def _load_scored_messages() -> pd.DataFrame:
    path = DATA_PROCESSED / "scored_messages.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Pipeline has not been run yet. Run `python -m src.pipeline`.")
    return pd.read_csv(path)


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

    user_risk_path = DATA_PROCESSED / "user_risk.csv"
    if not user_risk_path.exists():
        raise HTTPException(status_code=503, detail="Pipeline has not been run yet.")
    user_risk = pd.read_csv(user_risk_path)
    row = user_risk[user_risk["user"] == pseudonym]
    if row.empty:
        raise HTTPException(status_code=404, detail="User pseudonym not found.")

    summary = row.iloc[0].to_dict()
    explanation = explain_drift(pseudonym, summary)

    log_event("watsonx_explanation_generated", pseudonym,
              {"explanation_generated": explanation is not None}, actor=role)

    return {
        "user_pseudonym": pseudonym,
        "summary": summary,
        "explanation": explanation or "watsonx.ai not configured — set WATSONX_API_KEY/WATSONX_PROJECT_ID in .env.",
    }
