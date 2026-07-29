"""
Orchestrates the full PART 1 + PART 2 pipeline end-to-end and writes
results to data/processed/ so the API can serve them without recomputing
on every request. Run this after dropping a new dataset into data/raw/,
or on a schedule in production.

    python -m src.pipeline
"""
import logging
import pandas as pd

from src.utils import allow_unverified_ssl_for_nltk_downloads
allow_unverified_ssl_for_nltk_downloads()

from src.config import DATA_PROCESSED
from src.data.ingest import load_email_data, load_insider_labels
from src.data.anonymize import anonymize_dataframe, pseudonymize_user
from src.data.validate import validate_email_df
from src.features.linguistic_features import extract_features_df
from src.features.stylometry import extract_stylometry_df
from src.features.baseline_engine import BaselineEngine
from src.features.drift_scoring import score_drift_df, aggregate_user_risk
from src.models.unsupervised.anomaly_detection import fit_isolation_forest, fit_dbscan_clusters
from src.governance.audit_log import log_event
from src.governance.bias_audit import audit_drift_by_style_proxy
from src.governance.openscale_monitor import build_monitor_snapshot, compare_to_previous_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline() -> dict:
    logger.info("=== Starting pipeline run ===")

    raw = load_email_data()
    validate_email_df(raw)

    anon = anonymize_dataframe(raw)
    feats = extract_features_df(anon)
    feats = extract_stylometry_df(feats)

    engine = BaselineEngine()
    z_df = engine.replay(feats)
    scored = score_drift_df(z_df)

    user_risk = aggregate_user_risk(scored)

    _, _, msg_with_anomaly = fit_isolation_forest(scored)
    _, _, user_with_clusters = fit_dbscan_clusters(user_risk)

    # persist for the API layer to read without recomputation
    scored.to_csv(DATA_PROCESSED / "scored_messages.csv", index=False)
    user_with_clusters.to_csv(DATA_PROCESSED / "user_risk.csv", index=False)

    labels = load_insider_labels()
    if labels is not None:
        labels = labels.copy()
        labels["user"] = labels["user"].map(pseudonymize_user)
        labels.to_csv(DATA_PROCESSED / "user_labels_pseudonymized.csv", index=False)

    top_flagged = user_with_clusters.sort_values("avg_drift_score", ascending=False).head(10)
    for _, row in top_flagged.iterrows():
        log_event(
            "drift_flagged",
            row["user"],
            {"avg_drift_score": row["avg_drift_score"], "flagged_message_rate": row["flagged_message_rate"]},
        )

    # Governance addenda: a per-run fairness check on writing-style proxies,
    # and an org-wide monitoring snapshot (feature-distribution drift over
    # time, independent of any single user) — see governance/bias_audit.py
    # and governance/openscale_monitor.py for what each actually checks.
    try:
        bias_report = audit_drift_by_style_proxy(scored)
        if bias_report["proxies_with_disparity_over_1_5x"]:
            logger.warning(
                f"Bias audit flagged possible disparity in: "
                f"{bias_report['proxies_with_disparity_over_1_5x']} — see data/processed/reports/bias_audit.md"
            )
    except Exception as e:
        logger.warning(f"Bias audit step failed, continuing without it: {e}")

    try:
        snapshot = build_monitor_snapshot(scored, user_with_clusters)
        comparison = compare_to_previous_snapshot(snapshot)
        if comparison["alerts"]:
            logger.warning(f"Monitoring snapshot flagged org-wide feature drift: {comparison['alerts']}")
    except Exception as e:
        logger.warning(f"Monitoring snapshot step failed, continuing without it: {e}")

    logger.info(f"=== Pipeline run complete: {len(scored)} messages, {len(user_risk)} users ===")
    return {
        "n_messages": len(scored),
        "n_users": len(user_risk),
        "processed_dir": str(DATA_PROCESSED),
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(result)