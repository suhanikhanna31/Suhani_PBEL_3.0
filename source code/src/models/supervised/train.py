"""
PART 2 — Supervised risk classifier.

Trains a RandomForest and an XGBoost classifier on per-user aggregated
features (drift scores + raw linguistic/stylometry aggregates) against
ground-truth insider labels (data/raw/insider_labels.csv when using the
synthetic generator; supply your own label file at the same path/schema
if using the real CERT dataset — CERT ships separate "answers" files
identifying synthetic insider scenarios you can map to this schema).

Both models are trained so the write-up can compare a bagging vs. a
boosting approach — a natural thing to discuss in an internship review.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import DATA_PROCESSED, ROOT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = ROOT_DIR / "src" / "models" / "supervised" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = DATA_PROCESSED / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "avg_drift_score", "max_drift_score", "n_messages",
    "n_flagged_messages", "flagged_message_rate",
]


def build_training_table(user_risk_df: pd.DataFrame, labels_df: pd.DataFrame,
                          user_col_risk: str = "user", user_col_labels: str = "user") -> pd.DataFrame:
    """
    Joins per-user aggregated risk/drift features (from drift_scoring.aggregate_user_risk)
    with ground-truth labels. NOTE: labels_df has *raw* usernames while
    user_risk_df has pseudonyms post-anonymize — callers must join on a
    consistent key (either both raw, pre-anonymization, or map labels
    through the same pseudonymize_user() function). See train.py entrypoint.
    """
    merged = user_risk_df.merge(labels_df, left_on=user_col_risk, right_on=user_col_labels, how="inner")
    return merged


def train_models(training_df: pd.DataFrame, label_col: str = "is_malicious_insider"):
    X = training_df[FEATURE_COLS].fillna(0.0)
    y = training_df[label_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    results = {}

    # --- RandomForest (bagging) ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else np.zeros(len(y_test))
    results["random_forest"] = _evaluate(rf, X_test, y_test, rf_proba, "RandomForest")

    # --- XGBoost (boosting) ---
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb_clf.fit(X_train, y_train)
    xgb_proba = xgb_clf.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else np.zeros(len(y_test))
    results["xgboost"] = _evaluate(xgb_clf, X_test, y_test, xgb_proba, "XGBoost")

    joblib.dump(rf, ARTIFACT_DIR / "random_forest.pkl")
    joblib.dump(xgb_clf, ARTIFACT_DIR / "xgboost.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    logger.info(f"Saved model artifacts to {ARTIFACT_DIR}")

    write_metrics_report(results, n_train=len(y_train), n_test=len(y_test),
                          n_positive=int(y.sum()), n_total=len(y))

    try:
        from src.models.supervised.explainability import write_shap_summary
        write_shap_summary(rf, X_test, FEATURE_COLS, model_name="random_forest")
        write_shap_summary(xgb_clf, X_test, FEATURE_COLS, model_name="xgboost")
    except ImportError:
        logger.warning("shap not installed — skipping feature-attribution report. `pip install shap` to enable.")
    except Exception as e:
        logger.warning(f"SHAP explainability step failed, continuing without it: {e}")

    return results


def write_metrics_report(results: dict, n_train: int, n_test: int,
                          n_positive: int, n_total: int) -> None:
    """
    Persists real, numeric evaluation results (not just qualitative claims)
    to data/processed/reports/, in both machine-readable JSON and a
    human-readable Markdown summary suitable for pasting into the project
    report or README.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "dataset": {
            "n_total_users": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "n_positive_class": n_positive,
            "class_balance": round(n_positive / n_total, 4) if n_total else None,
        },
        "models": {},
    }

    lines = [
        f"# Model evaluation report",
        f"",
        f"Generated: {timestamp}",
        f"",
        f"Dataset: {n_total} users ({n_positive} positive / insider) — "
        f"{n_train} train / {n_test} test rows.",
        f"",
        f"| Model | AUC | Precision (1) | Recall (1) | F1 (1) |",
        f"|---|---|---|---|---|",
    ]

    for name, res in results.items():
        report = res["report"]
        pos = report.get("1", {})
        auc = res["auc"]
        payload["models"][name] = {
            "auc": None if auc != auc else round(auc, 4),
            "precision_positive_class": round(pos.get("precision", 0.0), 4),
            "recall_positive_class": round(pos.get("recall", 0.0), 4),
            "f1_positive_class": round(pos.get("f1-score", 0.0), 4),
            "accuracy": round(report.get("accuracy", 0.0), 4),
            "confusion_matrix": res["confusion_matrix"],
        }
        auc_str = "n/a" if auc != auc else f"{auc:.3f}"
        lines.append(
            f"| {name} | {auc_str} | {pos.get('precision', 0.0):.3f} | "
            f"{pos.get('recall', 0.0):.3f} | {pos.get('f1-score', 0.0):.3f} |"
        )

    lines.append("")
    lines.append("Confusion matrices (rows = actual, cols = predicted, class order [0, 1]):")
    lines.append("")
    for name, res in results.items():
        lines.append(f"- **{name}**: `{res['confusion_matrix']}`")
    lines.append("")
    lines.append(
        "> These numbers come from a single held-out split on the current "
        "dataset (synthetic unless a real CERT file + labels are present) "
        "and are regenerated every time `train_models` runs — they are not "
        "hand-picked or hard-coded."
    )

    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2))
    (REPORTS_DIR / "model_metrics.md").write_text("\n".join(lines))
    logger.info(f"Wrote metrics report to {REPORTS_DIR}/model_metrics.{{json,md}}")


def _evaluate(model, X_test, y_test, proba, name):
    preds = (proba >= 0.5).astype(int)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    try:
        auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_test, preds).tolist()
    logger.info(f"[{name}] AUC={auc:.3f}" if auc == auc else f"[{name}] AUC=n/a (too few positive samples)")
    return {"report": report, "auc": auc, "confusion_matrix": cm}


def predict_risk_proba(model_name: str, X: pd.DataFrame) -> np.ndarray:
    """Load a saved model + scaler and return risk probabilities for new user-feature rows."""
    model = joblib.load(ARTIFACT_DIR / f"{model_name}.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    X_scaled = scaler.transform(X[FEATURE_COLS].fillna(0.0))
    return model.predict_proba(X_scaled)[:, 1]

if __name__ == "__main__":
    # Full pipeline smoke test using synthetic data end-to-end.
    from src.data.ingest import load_email_data, load_insider_labels
    from src.data.anonymize import anonymize_dataframe, pseudonymize_user
    from src.features.linguistic_features import extract_features_df
    from src.features.stylometry import extract_stylometry_df
    from src.features.baseline_engine import BaselineEngine
    from src.features.drift_scoring import score_drift_df, aggregate_user_risk

    raw = load_email_data()
    labels = load_insider_labels()

    if labels is None:
        # Supervised training has a hard dependency on ground-truth labels —
        # unlike watsonx/transformers/spaCy, there's no meaningful fallback
        # here. This happens whenever data/raw/ doesn't include a matching
        # insider_labels.csv, e.g. a real (unlabeled) CERT sample dropped in
        # to replace the synthetic generator (see README, "Model evaluation").
        # Exit cleanly rather than crashing, so this is distinguishable from
        # an actual bug in CI/local runs.
        print(
            "No insider_labels.csv found in data/raw/ — skipping supervised "
            "training (drift scoring and the unsupervised models don't need "
            "labels and are unaffected). Supply data/raw/insider_labels.csv "
            "with columns matching load_insider_labels()'s expected schema "
            "to enable this step."
        )
        raise SystemExit(0)

    anon = anonymize_dataframe(raw)
    feats = extract_features_df(anon)
    feats = extract_stylometry_df(feats)

    engine = BaselineEngine()
    z_df = engine.replay(feats)
    scored = score_drift_df(z_df)
    user_risk = aggregate_user_risk(scored)

    # map raw label usernames -> same pseudonyms used above
    labels["user"] = labels["user"].map(pseudonymize_user)
    training_table = build_training_table(user_risk, labels)

    results = train_models(training_table)
    print(results["random_forest"]["report"]["accuracy"], results["xgboost"]["report"]["accuracy"])
