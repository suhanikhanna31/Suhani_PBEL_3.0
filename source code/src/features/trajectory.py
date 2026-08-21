"""
PART 2 (extension) — Drift trajectory: rate-of-change, not just threshold.

drift_scoring.py answers "how far from normal is this message/user right
now" — a level. It says nothing about *how that level got there*. Someone
whose drift score has crept up slowly over three months reads very
differently from someone who spiked the same amount in a week, and a
pure-threshold system treats them identically the moment they both cross
DRIFT_Z_THRESHOLD. UEBA vendors talk about rate-of-change-aware alerting
regularly; few actually ship it, because a naive version (e.g. simple
message-to-message delta) is noisy — a single unusually calm or unusually
urgent message swings it wildly.

This module fits a linear trend (numpy.polyfit, degree 1) over each user's
recent chronological drift-score sequence instead of differencing
consecutive points, which is far less sensitive to single-message noise —
the slope is a least-squares fit across the whole recent window, not a
two-point comparison. It then compares the trend slope of the two halves
of that window to get a coarse curvature/acceleration signal: is the
*rate* of drift itself increasing, not just the drift.

Pure numpy, no new dependency — consistent with the option's own framing
("numpy only") and this project's existing preference (see
sliding_window.py, drift_scoring.py) for auditable arithmetic over an
opaque model wherever the two are substitutable.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Need at least this many chronological drift scores for a trend line to
# mean anything; below this, trend_slope is reported as 0.0 / not-yet-measurable
# rather than a noisy fit over 2-3 points.
MIN_POINTS_FOR_TREND = 5

# A trend_slope at or above this (drift-score units per message) alongside a
# positive acceleration is what flags "accelerating", not level alone.
ACCELERATION_THRESHOLD = 0.02


def _fit_slope(y: np.ndarray) -> float:
    """Least-squares linear trend slope over evenly-spaced x = 0..n-1. O(n)."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    # polyfit degree 1 -> [slope, intercept]; guard the rare all-identical-y
    # case (numpy warns/returns 0 slope there anyway, but stay explicit).
    if np.allclose(y, y[0]):
        return 0.0
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def compute_user_trajectory(drift_scores: np.ndarray) -> dict:
    """
    drift_scores: one user's drift_score values, in chronological order
    (oldest first). Returns the whole-window trend slope plus a coarse
    acceleration signal (second-half slope minus first-half slope).
    """
    n = len(drift_scores)
    if n < MIN_POINTS_FOR_TREND:
        return {"trend_slope": 0.0, "acceleration": 0.0, "accelerating": False, "n_points": n}

    whole_slope = _fit_slope(drift_scores)

    mid = n // 2
    first_half_slope = _fit_slope(drift_scores[:mid]) if mid >= 2 else 0.0
    second_half_slope = _fit_slope(drift_scores[mid:]) if (n - mid) >= 2 else 0.0
    acceleration = second_half_slope - first_half_slope

    accelerating = bool(whole_slope >= ACCELERATION_THRESHOLD and acceleration > 0)

    return {
        "trend_slope": round(whole_slope, 5),
        "acceleration": round(acceleration, 5),
        "accelerating": accelerating,
        "n_points": n,
    }


def compute_trajectory_df(
    scored_df: pd.DataFrame,
    user_col: str = "user",
    date_col: str = "date",
    score_col: str = "drift_score",
) -> pd.DataFrame:
    """
    Per-user trajectory summary from a chronologically-sortable scored-messages
    DataFrame. Returns one row per user with trend_slope / acceleration /
    accelerating — meant to be merged onto the per-user risk table
    (aggregate_user_risk() output in drift_scoring.py) alongside the existing
    level-based columns, not to replace them.
    """
    if date_col in scored_df.columns:
        ordered = scored_df.sort_values([user_col, date_col])
    else:
        ordered = scored_df.sort_values([user_col])

    logger.info(f"Fitting drift trajectories for {ordered[user_col].nunique()} users...")

    rows = []
    for user, group in ordered.groupby(user_col):
        traj = compute_user_trajectory(group[score_col].to_numpy(dtype=float))
        rows.append({user_col: user, **traj})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # slow creep vs. sharp spike, both ending at roughly the same level
    slow_creep = np.linspace(0.1, 0.9, 20)
    sharp_spike = np.concatenate([np.full(15, 0.1), np.linspace(0.1, 0.9, 5)])
    print("slow creep:", compute_user_trajectory(slow_creep))
    print("sharp spike:", compute_user_trajectory(sharp_spike))
