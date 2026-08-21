"""
Webhook consumed by a watsonx Assistant action, so an analyst can ask
things like "show me users whose communication tone changed significantly
this week" in natural language and have the Assistant call back into this
API. Configure the Assistant action's webhook URL to POST here.

Auth: shared-secret header (ASSISTANT_WEBHOOK_SECRET) — swap for IBM
Cloud IAM-based service-to-service auth in production.
"""
import logging

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from src.config import ASSISTANT_WEBHOOK_SECRET, TOP_K_RISKIEST
from src.data.store import read_df
from src.dsa.top_k_heap import TopKRiskHeap
from src.governance.access_control import get_current_role, check_permission
from src.governance.audit_log import log_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


class AssistantQuery(BaseModel):
    intent: str  # e.g. "top_risky_users", "user_summary"
    parameters: dict = {}


def _verify_secret(x_webhook_secret: str):
    if x_webhook_secret != ASSISTANT_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret.")


@router.post("/webhook")
def assistant_webhook(query: AssistantQuery, x_webhook_secret: str = Header(default=None)):
    _verify_secret(x_webhook_secret)

    df = read_df("user_risk")
    if df is None:
        return {"assistant_response": "The pipeline hasn't been run yet — no data available."}

    if query.intent == "top_risky_users":
        k = int(query.parameters.get("k", TOP_K_RISKIEST))
        heap = TopKRiskHeap(k=k)
        for _, row in df.iterrows():
            heap.push(row["user"], float(row["avg_drift_score"]))
        top = heap.top_k()
        lines = [f"{e.user_id}: drift score {e.score:.2f}" for e in top]
        return {
            "assistant_response": f"Top {len(top)} users by communication drift this period:\n" + "\n".join(lines),
            "data": [{"user_pseudonym": e.user_id, "avg_drift_score": e.score} for e in top],
        }

    if query.intent == "investigate_user":
        pseudonym = query.parameters.get("user_pseudonym")
        from src.agents.investigation_agent import investigate_user
        report = investigate_user(pseudonym, actor="watsonx_assistant")
        if report.get("error"):
            return {"assistant_response": report["error"]}
        return {
            "assistant_response": (
                f"Investigation for {pseudonym}: recommended action is "
                f"'{report['recommendation']}'. {report['rationale']} "
                "This is a recommendation only — please review before acting."
            ),
            "data": report,
        }

    if query.intent == "user_summary":
        pseudonym = query.parameters.get("user_pseudonym")
        row = df[df["user"] == pseudonym]
        if row.empty:
            return {"assistant_response": f"No data found for {pseudonym}."}
        r = row.iloc[0]
        return {
            "assistant_response": (
                f"{pseudonym}: avg drift score {r['avg_drift_score']:.2f}, "
                f"{int(r['n_flagged_messages'])} of {int(r['n_messages'])} messages flagged "
                f"({r['flagged_message_rate']*100:.1f}%)."
            ),
            "data": r.to_dict(),
        }

    return {"assistant_response": f"Unrecognized intent '{query.intent}'."}


class AskQuery(BaseModel):
    question: str
    k: int = 4


@router.post("/ask")
def assistant_ask(query: AskQuery, x_webhook_secret: str = Header(default=None)):
    """
    Open-ended RAG endpoint, complementary to /webhook's structured
    intents above. Where /webhook answers questions about *risk data*
    ("who's riskiest right now"), /ask answers questions about *how the
    system itself works* — consent, anonymization, audit logging,
    architecture — retrieved from this project's own docs/audit log
    (src/governance/rag.py) rather than the risk data itself. Same auth
    as /webhook for now; a dashboard-facing UI would sit this behind
    proper session auth instead of the shared secret.
    """
    _verify_secret(x_webhook_secret)

    from src.governance.rag import answer_question
    result = answer_question(query.question, k=query.k)
    return {
        "assistant_response": result["answer"],
        "sources": result["sources"],
        "watsonx_generated": result["watsonx_generated"],
    }


@router.post("/dashboard-ask")
def dashboard_ask(query: AskQuery, role: str = Depends(get_current_role)):
    """
    Dashboard-facing counterpart to /ask above, same underlying RAG call
    but authenticated the same way the rest of the dashboard is (role
    header via get_current_role/check_permission) instead of the
    watsonx-Assistant-to-backend webhook secret — the frontend's "Ask
    about this system" panel calls this route directly from the browser,
    where there's no shared secret to send. Was previously missing
    entirely: the frontend called POST /api/assistant/dashboard-ask,
    nothing matched it, and the request fell through to the StaticFiles
    catch-all mount (GET/HEAD only) — hence the "Method Not Allowed" the
    dashboard showed instead of a real answer.
    """
    check_permission(role, "ask_assistant")

    from src.governance.rag import answer_question
    result = answer_question(query.question, k=query.k)

    log_event("dashboard_ask", "n/a",
              {"question": query.question, "watsonx_generated": result["watsonx_generated"]}, actor=role)

    return {
        "assistant_response": result["answer"],
        "sources": result["sources"],
        "watsonx_generated": result["watsonx_generated"],
    }
