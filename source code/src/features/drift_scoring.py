"""
PART 2 — Drift scoring.

Turns the per-feature z-scores from baseline_engine.py into a single
interpretable drift score per message/user, using the config threshold
(DRIFT_Z_THRESHOLD) to flag which features actually crossed into
"significant drift" territory.

This is intentionally simple and auditable (a weighted absolute z-score
average + a flagged-feature count) rather than a black box, because the
governance/ethics angle of this project depends on analysts being able to
see *why* a user was flagged, not just a bare score
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
#
# social_engineering_score weighted at 1.8, just under urgency_score (2.0):
# it's a targeted-manipulation signal (authority spoofing, isolation framing,
# etc.) rather than a general tone shift, so it's treated as similarly
# diagnostic to urgency rather than left at the 1.0 default a brand-new,
# uncalibrated feature would otherwise silently get.
FEATURE_WEIGHTS = {
    "z_urgency_score": 2.0,
    "z_social_engineering_score": 1.8,
    "z_sentiment_polarity": 1.5,
    "z_sentiment_subjectivity": 1.0,
    "z_function_word_ratio": 1.2,
    "z_lexical_diversity": 1.0,
    "z_readability_flesch": 0.8,
    "z_noun_ratio": 0.7,
    "z_verb_ratio": 0.7,
    # Communication-network drift (src/dsa/communication_graph.py). Weighted
    # close to sentiment/urgency rather than at the 1.0 default: a shrinking,
    # concentrated circle is one of the best-documented pre-incident
    # behaviors in real counterintelligence casework, not a minor stylistic
    # wobble, so it's treated as similarly diagnostic to tone/urgency drift.
    "z_contact_degree": 1.4,
    "z_interaction_concentration": 1.6,
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


def feature_contributions(z_row: Dict[str, float]) -> List[dict]:
    """
    Same weighted-|z| math as score_drift() above, but returns the
    per-feature breakdown instead of collapsing it into one number —
    this is what makes the drift score auditable rather than a black box:
    every feature's exact share of a given score, not just the total.

    Returns a list of {feature, z, weight, contribution_pct}, sorted by
    contribution_pct descending. contribution_pct values sum to ~100 (modulo
    rounding) and are computed with the exact same clipped/weighted terms
    score_drift() uses, so they always explain the *actual* displayed
    drift_score for that row — never an approximation of it.
    """
    if not z_row:
        return []

    terms = []
    weighted_total = 0.0
    for feat, z in z_row.items():
        if pd.isna(z):
            continue
        w = FEATURE_WEIGHTS.get(feat, 1.0)
        clipped_z = min(abs(z), MAX_ABS_Z)
        weighted = w * clipped_z
        weighted_total += weighted
        terms.append({"feature": feat.replace("z_", ""), "z": round(float(z), 3), "weight": float(w), "_weighted": float(weighted)})

    if weighted_total <= 0:
        return [{"feature": t["feature"], "z": t["z"], "weight": t["weight"], "contribution_pct": 0.0} for t in terms]

    out = []
    for t in terms:
        pct = round(100 * t["_weighted"] / weighted_total, 1)
        out.append({"feature": t["feature"], "z": t["z"], "weight": t["weight"], "contribution_pct": float(pct)})
    out.sort(key=lambda t: t["contribution_pct"], reverse=True)
    return out


def score_drift_df(z_df: pd.DataFrame) -> pd.DataFrame:
    """Apply score_drift row-wise across a DataFrame that already has z_* columns."""
    z_cols = [c for c in z_df.columns if c.startswith("z_")]
    logger.info(f"Scoring drift across {len(z_df)} rows using {len(z_cols)} z-score features...")

    if not z_cols:
        # No user in this batch has enough prior history yet to fill even one
        # feature's baseline window (see BaselineEngine.is_ready) — e.g. a
        # short/sparse real-data sample where nobody reaches
        # BASELINE_WINDOW_SIZE messages. pandas' .apply(axis=1) over a
        # zero-column DataFrame doesn't reliably reproduce a per-row Series of
        # dicts, so handle this case explicitly instead of relying on it.
        logger.warning(
            "No z-score columns present — no user in this batch has enough "
            "message history yet to fill a baseline window. Every message is "
            "being scored as drift_score=0.0 (not yet measurable), not as "
            "'no drift detected'."
        )
        result_df = pd.DataFrame(
            {
                "drift_score": 0.0,
                "flagged_features": [[] for _ in range(len(z_df))],
                "n_flagged": 0,
            },
            index=z_df.index,
        )
        return pd.concat([z_df, result_df], axis=1)

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
    sample = {"z_urgency_score": 3.1, "z_social_engineering_score": 2.6, "z_sentiment_polarity": -2.8, "z_readability_flesch": 0.4}
    print(score_drift(sample))
