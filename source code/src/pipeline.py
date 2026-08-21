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
from src.data.store import write_df
from src.data.anonymize import anonymize_dataframe, pseudonymize_user
from src.data.validate import validate_email_df
from src.features.linguistic_features import extract_features_df
from src.features.stylometry import extract_stylometry_df
from src.features.network_features import extract_network_features_df
from src.features.baseline_engine import BaselineEngine
from src.features.drift_scoring import score_drift_df, aggregate_user_risk
from src.features.trajectory import compute_trajectory_df
from src.models.unsupervised.anomaly_detection import fit_isolation_forest, fit_dbscan_clusters
from src.governance.audit_log import log_event
from src.governance.bias_audit import audit_drift_by_style_proxy
from src.governance.openscale_monitor import build_monitor_snapshot, compare_to_previous_snapshot
from src.governance.alert_suppression import apply_context_aware_suppression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline() -> dict:
    logger.info("=== Starting pipeline run ===")

    raw = load_email_data()
    validate_email_df(raw)

    # Network-drift features (contact_degree, interaction_concentration) are
    # computed from raw to/cc/bcc *before* anonymization, keyed to raw's own
    # row index: anonymize_dataframe()'s single-string HMAC over the whole
    # semicolon-joined `to` field is meant for the single-address `user`/
    # `from` columns, not a multi-recipient list, so computing network
    # features first and re-aligning by index afterward avoids depending on
    # that column post-anonymization. Only the two resulting numeric columns
    # are carried forward — no raw address ever leaves this block.
    net_feats = extract_network_features_df(raw, user_col="user", date_col="date")
    net_cols = net_feats[["contact_degree", "interaction_concentration"]]

    anon = anonymize_dataframe(raw)
    anon = anon.join(net_cols)  # index-aligned; consent-filtered rows keep only their own values
    feats = extract_features_df(anon)
    feats = extract_stylometry_df(feats)

    engine = BaselineEngine()
    z_df = engine.replay(feats)
    scored = score_drift_df(z_df)

    user_risk = aggregate_user_risk(scored)

    # Trajectory: fits each user's own chronological drift-score sequence
    # (numpy.polyfit) to get trend_slope + acceleration, merged onto the
    # existing level-based risk table rather than replacing it — see
    # src/features/trajectory.py.
    trajectory = compute_trajectory_df(scored, user_col="user", date_col="date")
    user_risk = user_risk.merge(trajectory, on="user", how="left")

    _, _, msg_with_anomaly = fit_isolation_forest(scored)
    _, _, user_with_clusters = fit_dbscan_clusters(user_risk)

    # Org-wide monitoring snapshot is computed here (rather than only in the
    # governance-addenda block below) because context-aware alert
    # suppression needs to know *before* persisting user_risk whether this
    # run shows a correlated, organization-wide shift — see
    # governance/alert_suppression.py. Kept in its own try/except, same
    # graceful-degradation pattern as the rest of this function: a snapshot
    # failure should never block the core scoring pipeline from persisting.
    org_wide_shift_detected = False
    snapshot = None
    try:
        snapshot = build_monitor_snapshot(scored, user_with_clusters)
        snapshot_comparison = compare_to_previous_snapshot(snapshot)
        org_wide_shift_detected = bool(snapshot_comparison["alerts"])
        if org_wide_shift_detected:
            logger.warning(f"Monitoring snapshot flagged org-wide feature drift: {snapshot_comparison['alerts']}")
    except Exception as e:
        logger.warning(f"Monitoring snapshot step failed, continuing without it: {e}")

    # Context-aware alert suppression: gates individual escalation on
    # relative standing among this run's own (possibly also-elevated)
    # population, raising the bar further when the whole org shifted
    # together — see governance/alert_suppression.py.
    user_with_clusters = apply_context_aware_suppression(
        user_with_clusters, org_wide_shift_detected=org_wide_shift_detected
    )

    # persist for the API layer to read without recomputation
    # NOTE: must write msg_with_anomaly (not the original `scored`) — it's the
    # frame with IsolationForest's anomaly_score/is_anomaly columns attached.
    # Writing `scored` here silently dropped those columns before they ever
    # reached the API, even though the model ran and logged its results above.
    #
    # write_df (src/data/store.py) writes to the live Neon `scored_messages`/
    # `user_risk` tables when DATABASE_URL is set — not just a local CSV —
    # since a deployed API (e.g. on Vercel) has no persistent local disk to
    # read data/processed/*.csv back from between requests or deploys.
    write_df(msg_with_anomaly, "scored_messages")
    write_df(user_with_clusters, "user_risk")

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

    # Governance addendum: a per-run fairness check on writing-style proxies
    # — see governance/bias_audit.py. (The org-wide monitoring snapshot is
    # computed earlier now, since alert suppression depends on it — see
    # above.)
    try:
        bias_report = audit_drift_by_style_proxy(scored)
        if bias_report["proxies_with_disparity_over_1_5x"]:
            logger.warning(
                f"Bias audit flagged possible disparity in: "
                f"{bias_report['proxies_with_disparity_over_1_5x']} — see data/processed/reports/bias_audit.md"
            )
    except Exception as e:
        logger.warning(f"Bias audit step failed, continuing without it: {e}")

    logger.info(f"=== Pipeline run complete: {len(scored)} messages, {len(user_risk)} users ===")
    return {
        "n_messages": len(scored),
        "n_users": len(user_risk),
        "processed_dir": str(DATA_PROCESSED),
        "org_wide_shift_detected": org_wide_shift_detected,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(result)
