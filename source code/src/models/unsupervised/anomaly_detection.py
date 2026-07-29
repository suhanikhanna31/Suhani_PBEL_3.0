"""
PART 2 — Unsupervised drift clustering / anomaly detection.

Supervised labels (data/raw/insider_labels.csv) only exist because the
CERT dataset was built with synthetic ground truth baked in. Real
deployments won't have reliable labels for novel insider behavior — so
this module provides a label-free complement:

- IsolationForest: flags individual messages/users as outliers in
  feature space without needing labels. Good for "this looks weird"
  scoring at message level.
- DBSCAN: clusters users by their aggregated linguistic/behavioral
  profile; users who don't fall into any dense cluster (label == -1)
  are structurally different from their peers — a distinct signal from
  "high drift from their own baseline."

Both outputs get merged with the supervised risk score downstream so an
analyst sees three independent signals, not one black-box number.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import ROOT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = ROOT_DIR / "src" / "models" / "unsupervised" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MESSAGE_FEATURE_COLS = [
    "sentiment_polarity", "sentiment_subjectivity", "urgency_score",
    "readability_flesch", "lexical_diversity", "function_word_ratio",
    "noun_ratio", "verb_ratio", "caps_ratio", "exclamation_ratio",
]

USER_FEATURE_COLS = [
    "avg_drift_score", "max_drift_score", "flagged_message_rate",
]


def fit_isolation_forest(message_df: pd.DataFrame, contamination: float = 0.05):
    """Message-level anomaly detection. contamination = expected outlier fraction."""
    X = message_df[MESSAGE_FEATURE_COLS].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)  # higher = more normal
    preds = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

    joblib.dump(model, ARTIFACT_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "if_scaler.pkl")

    result = message_df.copy()
    result["anomaly_score"] = -scores  # flip sign so higher = more anomalous, consistent w/ drift_score
    result["is_anomaly"] = (preds == -1)
    logger.info(f"IsolationForest flagged {result['is_anomaly'].sum()}/{len(result)} messages as anomalous.")
    return model, scaler, result


def fit_dbscan_clusters(user_risk_df: pd.DataFrame, eps: float = 0.8, min_samples: int = 3):
    """
    User-level structural clustering. Users assigned cluster == -1 are
    "noise" points — they don't resemble any dense group of peers, which
    is itself a signal worth surfacing (distinct from within-user drift).
    """
    X = user_risk_df[USER_FEATURE_COLS].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    joblib.dump(scaler, ARTIFACT_DIR / "dbscan_scaler.pkl")

    result = user_risk_df.copy()
    result["cluster"] = labels
    result["is_outlier_cluster"] = (labels == -1)
    n_outliers = result["is_outlier_cluster"].sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"DBSCAN found {n_clusters} clusters; {n_outliers} users are structural outliers.")
    return model, scaler, result


if __name__ == "__main__":
    from src.data.ingest import load_email_data
    from src.data.anonymize import anonymize_dataframe
    from src.features.linguistic_features import extract_features_df
    from src.features.stylometry import extract_stylometry_df
    from src.features.baseline_engine import BaselineEngine
    from src.features.drift_scoring import score_drift_df, aggregate_user_risk

    raw = load_email_data()
    anon = anonymize_dataframe(raw)
    feats = extract_features_df(anon)
    feats = extract_stylometry_df(feats)

    _, _, msg_result = fit_isolation_forest(feats)

    engine = BaselineEngine()
    z_df = engine.replay(feats)
    scored = score_drift_df(z_df)
    user_risk = aggregate_user_risk(scored)

    _, _, cluster_result = fit_dbscan_clusters(user_risk)
    print(cluster_result.sort_values("is_outlier_cluster", ascending=False).head())
