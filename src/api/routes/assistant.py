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
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.config import DATA_PROCESSED, ASSISTANT_WEBHOOK_SECRET, TOP_K_RISKIEST
from src.dsa.top_k_heap import TopKRiskHeap

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

    path = DATA_PROCESSED / "user_risk.csv"
    if not path.exists():
        return {"assistant_response": "The pipeline hasn't been run yet — no data available."}
    df = pd.read_csv(path)

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
