"""
PART 2 — Drift scoring.

Turns the per-feature z-scores from baseline_engine.py into a single
interpretable drift score per message/user, using the config threshold
(DRIFT_Z_THRESHOLD) to flag which features actually crossed into
"significant drift" territory.

This is intentionally simple and auditable (a weighted absolute z-score
average + a flagged-feature count) rather than a black box, because the
governance/ethics angle of this project depends on analysts being able to
see *why* a user was flagged, not just a bare score.
"""
import logging
from typing import Dict, List

import pandas as pd

from src.config import DRIFT_Z_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Guard rail: when a feature has near-zero variance in a user's baseline
# window, a single differing value can produce an enormous raw z-score
# (mathematically correct, but it would silently dominate the weighted
# average and drown out every other signal). We clip |z| before weighting
# so one near-constant feature can't blow up the overall drift score —
# it still crosses DRIFT_Z_THRESHOLD and gets flagged, it just can't
# multiply out to an uninterpretable magnitude.
MAX_ABS_Z = 10.0

# Not all features are equally diagnostic of insider-risk-relevant drift;
# urgency and sentiment shifts matter more than e.g. readability alone.
FEATURE_WEIGHTS = {
    "z_urgency_score": 2.0,
    "z_sentiment_polarity": 1.5,
    "z_sentiment_subjectivity": 1.0,
    "z_function_word_ratio": 1.2,
    "z_lexical_diversity": 1.0,
    "z_readability_flesch": 0.8,
    "z_noun_ratio": 0.7,
    "z_verb_ratio": 0.7,
}


def score_drift(z_row: Dict[str, float]) -> dict:
    """
    Given a dict of z_<feature> -> z-score values (as produced by
    BaselineEngine.replay), returns:
      - drift_score: weighted mean absolute z-score (0 = no drift; grows unbounded)
      - flagged_features: list of features whose |z| exceeded DRIFT_Z_THRESHOLD
      - n_flagged: convenience count
    """
    if not z_row:
        return {"drift_score": 0.0, "flagged_features": [], "n_flagged": 0}

    weighted_sum = 0.0
    weight_total = 0.0
    flagged: List[str] = []

    for feat, z in z_row.items():
        if pd.isna(z):
            continue
        w = FEATURE_WEIGHTS.get(feat, 1.0)
        clipped_z = min(abs(z), MAX_ABS_Z)
        weighted_sum += w * clipped_z
        weight_total += w
        if abs(z) >= DRIFT_Z_THRESHOLD:
            flagged.append(feat.replace("z_", ""))

    drift_score = round(weighted_sum / weight_total, 4) if weight_total else 0.0
    return {"drift_score": drift_score, "flagged_features": flagged, "n_flagged": len(flagged)}


def score_drift_df(z_df: pd.DataFrame) -> pd.DataFrame:
    """Apply score_drift row-wise across a DataFrame that already has z_* columns."""
    z_cols = [c for c in z_df.columns if c.startswith("z_")]
    logger.info(f"Scoring drift across {len(z_df)} rows using {len(z_cols)} z-score features...")

    results = z_df[z_cols].apply(lambda row: score_drift(row.dropna().to_dict()), axis=1)
    result_df = pd.DataFrame(list(results), index=z_df.index)
    return pd.concat([z_df, result_df], axis=1)


def aggregate_user_risk(scored_df: pd.DataFrame, user_col: str = "user") -> pd.DataFrame:
    """
    Rolls per-message drift scores up to a per-user risk summary — this is
    what feeds the Top-K heap (src/dsa/top_k_heap.py) for the analyst
    dashboard's "riskiest users this week" view.
    """
    agg = scored_df.groupby(user_col).agg(
        avg_drift_score=("drift_score", "mean"),
        max_drift_score=("drift_score", "max"),
        n_messages=("drift_score", "count"),
        n_flagged_messages=("n_flagged", lambda s: (s > 0).sum()),
    ).reset_index()
    agg["flagged_message_rate"] = (agg["n_flagged_messages"] / agg["n_messages"]).round(4)
    return agg.sort_values("avg_drift_score", ascending=False)


if __name__ == "__main__":
    sample = {"z_urgency_score": 3.1, "z_sentiment_polarity": -2.8, "z_readability_flesch": 0.4}
    print(score_drift(sample))
