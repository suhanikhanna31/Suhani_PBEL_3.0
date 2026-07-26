"""
PART 2 (addendum) — Feature-attribution explainability for the supervised
classifiers.

The rest of the project explains *drift* scores in plain terms (which
features moved, by how much, in drift_scoring.py) and via watsonx.ai's
natural-language summaries. This module adds the missing piece: an
explanation of the *classifier's own* decision — which aggregated
per-user features (avg_drift_score, max_drift_score, n_messages,
flagged_message_rate) actually drove a RandomForest/XGBoost prediction,
using SHAP (SHapley Additive exPlanations).

This is deliberately a thin, optional layer (train.py degrades gracefully
if `shap` isn't installed) so it doesn't become a new hard dependency for
a project whose core design principle is "run without every optional
extra configured."
"""
import logging
from pathlib import Path
from typing import List

import numpy as np

from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_PROCESSED / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def write_shap_summary(model, X_test, feature_names: List[str], model_name: str) -> None:
    """
    Computes SHAP values for `model` on `X_test`, saves a summary bar plot
    (mean |SHAP value| per feature) as a PNG, and a JSON file with the
    ranked feature importances — so "why did the model flag this user"
    has a real, model-grounded answer instead of only the drift score's
    own explanation.
    """
    import shap
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classifiers: shap_values comes back in one of a few shapes
    # depending on the sklearn/xgboost/shap version —
    #   - a list [class0_array, class1_array]                  (older shap)
    #   - a single 2D array (n_samples, n_features)             (xgboost, regression-style)
    #   - a single 3D array (n_samples, n_features, n_classes)  (newer shap + sklearn)
    # In every case we want the positive (class 1) contributions as a
    # plain 2D (n_samples, n_features) array.
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, -1]  # last class = positive class

    mean_abs = np.abs(values).mean(axis=0).reshape(-1)
    ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)

    (REPORTS_DIR / f"shap_importance_{model_name}.json").write_text(
        json.dumps([{"feature": f, "mean_abs_shap": round(v, 5)} for f, v in ranked], indent=2)
    )

    plt.figure(figsize=(6, 4))
    names = [f for f, _ in ranked]
    scores = [v for _, v in ranked]
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel("mean |SHAP value|")
    plt.title(f"Feature attribution — {model_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"shap_importance_{model_name}.png", dpi=120)
    plt.close()

    logger.info(f"Wrote SHAP feature-attribution report for {model_name} to {REPORTS_DIR}")
