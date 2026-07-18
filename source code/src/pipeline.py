"""
Orchestrates the full PART 1 + PART 2 pipeline end-to-end and writes
results to data/processed/ so the API can serve them without recomputing
on every request. Run this after dropping a new dataset into data/raw/,
or on a schedule in production.

    python -m src.pipeline
"""
import logging
import pandas as pd

# --- FIX: Global SSL bypass for NLTK dataset downloads ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ---------------------------------------------------------

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

    logger.info(f"=== Pipeline run complete: {len(scored)} messages, {len(user_risk)} users ===")
    return {
        "n_messages": len(scored),
        "n_users": len(user_risk),
        "processed_dir": str(DATA_PROCESSED),
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(result)