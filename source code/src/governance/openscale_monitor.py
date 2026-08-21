"""
watsonx.OpenScale-style continuous monitoring snapshot.

audit_log.py answers "did anything get tampered with" (integrity).
bias_audit.py answers "is any writing-style group flagged disproportionately"
(a one-time/ad-hoc fairness check).

This module adds the third leg IBM's real watsonx.OpenScale product
covers for deployed models: a periodic MONITORING SNAPSHOT — drift in the
*input feature distribution* itself (has average message urgency/
sentiment across the whole org shifted since last week, independent of
any one user?), and basic model-quality tracking over time (positive
rate, mean predicted risk). Run this on a schedule (e.g. daily/weekly, via
cron or a Code Engine scheduled job) alongside the main pipeline.

This is a lightweight, self-contained analogue built for a project scale
this can be fully audited within — not a client for the real
watsonx.OpenScale API. Wiring it up to OpenScale's actual REST API is a
credentials-and-endpoint swap once a project/instance exists; the
snapshot schema below is intentionally close to what OpenScale reports
(feature-level drift + basic performance stats) so that swap would be
mostly plumbing, not a redesign.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_PROCESSED / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_HISTORY_PATH = REPORTS_DIR / "openscale_monitor_history.jsonl"

FEATURE_DISTRIBUTION_COLS = [
    "sentiment_polarity", "sentiment_subjectivity", "urgency_score",
    "readability_flesch", "lexical_diversity",
]


def build_monitor_snapshot(scored_messages: pd.DataFrame, user_risk: pd.DataFrame) -> dict:
    """
    Summarizes the current run's feature distributions and aggregate risk
    output into a single snapshot record, and appends it to a JSONL
    history file so successive runs can be compared over time (org-wide
    distribution drift, not per-user drift).
    """
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_messages": int(len(scored_messages)),
        "n_users": int(len(user_risk)),
        "feature_distributions": {},
        "risk_summary": {
            "mean_avg_drift_score": float(user_risk["avg_drift_score"].mean()) if len(user_risk) else None,
            "pct_users_flagged_any": float((user_risk["n_flagged_messages"] > 0).mean()) if len(user_risk) else None,
            "top_decile_drift_threshold": float(user_risk["avg_drift_score"].quantile(0.9)) if len(user_risk) else None,
        },
    }

    for col in FEATURE_DISTRIBUTION_COLS:
        if col in scored_messages.columns:
            snapshot["feature_distributions"][col] = {
                "mean": float(scored_messages[col].mean()),
                "std": float(scored_messages[col].std()),
            }

    with open(MONITOR_HISTORY_PATH, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    logger.info(f"Appended monitoring snapshot to {MONITOR_HISTORY_PATH}")
    return snapshot


def compare_to_previous_snapshot(current: dict, z_alert_threshold: float = 2.0) -> dict:
    """
    Loads prior snapshots (if any) and flags any feature whose mean has
    shifted by more than `z_alert_threshold` standard deviations (using
    the *current* run's std as the reference scale) since the run before
    this one — a coarse, org-wide analogue of OpenScale's drift alerts,
    distinct from the per-user drift the main pipeline already computes.
    """
    if not MONITOR_HISTORY_PATH.exists():
        return {"previous_snapshot_available": False, "alerts": []}

    lines = [json.loads(l) for l in MONITOR_HISTORY_PATH.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return {"previous_snapshot_available": False, "alerts": []}

    previous = lines[-2]
    alerts = []
    for feat, stats in current.get("feature_distributions", {}).items():
        prev_stats = previous.get("feature_distributions", {}).get(feat)
        if not prev_stats or not stats.get("std"):
            continue
        shift = abs(stats["mean"] - prev_stats["mean"]) / max(stats["std"], 1e-9)
        if shift >= z_alert_threshold:
            alerts.append({"feature": feat, "shift_in_std_devs": round(shift, 3),
                            "previous_mean": prev_stats["mean"], "current_mean": stats["mean"]})

    return {"previous_snapshot_available": True, "alerts": alerts}


if __name__ == "__main__":
    scored_path = DATA_PROCESSED / "scored_messages.csv"
    risk_path = DATA_PROCESSED / "user_risk.csv"
    if not scored_path.exists() or not risk_path.exists():
        raise SystemExit("Run `python -m src.pipeline` first to produce scored_messages.csv / user_risk.csv")

    scored = pd.read_csv(scored_path)
    risk = pd.read_csv(risk_path)
    snap = build_monitor_snapshot(scored, risk)
    comparison = compare_to_previous_snapshot(snap)
    print(json.dumps({"snapshot": snap, "comparison": comparison}, indent=2))
