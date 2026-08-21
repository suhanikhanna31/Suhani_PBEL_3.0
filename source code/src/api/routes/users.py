"""
GET /api/users/risk        -> top-K riskiest users (Top-K heap, DSA-driven)
GET /api/users/{pseudonym} -> single user's risk detail
"""
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.config import TOP_K_RISKIEST
from src.data.store import read_df
from src.dsa.top_k_heap import TopKRiskHeap
from src.governance.access_control import get_current_role, check_permission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


def _load_user_risk() -> pd.DataFrame:
    df = read_df("user_risk")
    if df is None:
        raise HTTPException(status_code=503, detail="Pipeline has not been run yet. Run `python -m src.pipeline`.")
    return df


@router.get("/risk")
def top_risky_users(k: int = TOP_K_RISKIEST, role: str = Depends(get_current_role)):
    check_permission(role, "view_users")
    df = _load_user_risk()

    heap = TopKRiskHeap(k=k)
    for _, row in df.iterrows():
        heap.push(
            user_id=row["user"],
            score=float(row["avg_drift_score"]),
            details={
                "max_drift_score": row["max_drift_score"],
                "n_messages": int(row["n_messages"]),
                "flagged_message_rate": row["flagged_message_rate"],
                "is_outlier_cluster": bool(row.get("is_outlier_cluster", False)),
                # Trajectory (src/features/trajectory.py) — rate-of-change,
                # not just level.
                "trend_slope": row.get("trend_slope", 0.0),
                "accelerating": bool(row.get("accelerating", False)),
                # Context-aware alert suppression
                # (src/governance/alert_suppression.py) — relative standing
                # among this run's own population, gated harder during a
                # correlated org-wide event.
                "cohort_z_score": row.get("cohort_z_score", 0.0),
                "escalation_recommended": bool(row.get("escalation_recommended", False)),
                "suppressed_correlated_event": bool(row.get("suppressed_correlated_event", False)),
            },
        )
    return [
        {"user_pseudonym": e.user_id, "avg_drift_score": e.score, **e.details}
        for e in heap.top_k()
    ]


@router.get("/{pseudonym}")
def get_user_detail(pseudonym: str, role: str = Depends(get_current_role)):
    check_permission(role, "view_users")
    df = _load_user_risk()
    row = df[df["user"] == pseudonym]
    if row.empty:
        raise HTTPException(status_code=404, detail="User pseudonym not found.")
    return row.iloc[0].to_dict()
