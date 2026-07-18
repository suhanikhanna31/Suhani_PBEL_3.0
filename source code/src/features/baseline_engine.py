"""
PART 1 — Per-user baseline engine.

Maintains one SlidingWindowStats (src/dsa/sliding_window.py) per feature,
per user — e.g. user emp_abc123's baseline for sentiment_polarity is a
window of their last N (config.BASELINE_WINDOW_SIZE) messages. As new
messages arrive we push into the relevant window; drift_scoring.py then
compares a "current" value against that window's mean/std via z-score.

Users' baselines are looked up through an LRU cache (src/dsa/lru_cache_baselines.py)
so we're not holding every user's full baseline state in memory forever
in a long-running service — only the recently-active ones.
"""
import logging
from typing import Dict

import pandas as pd

from src.config import BASELINE_WINDOW_SIZE
from src.dsa.sliding_window import SlidingWindowStats
from src.dsa.lru_cache_baselines import LRUCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The features we track drift on. Keep this list in sync with what
# linguistic_features.py / stylometry.py actually produce.
TRACKED_FEATURES = [
    "sentiment_polarity", "sentiment_subjectivity", "urgency_score",
    "readability_flesch", "lexical_diversity", "function_word_ratio",
    "noun_ratio", "verb_ratio",
]


class UserBaseline:
    """All tracked-feature sliding windows for one user."""

    def __init__(self, window_size: int = BASELINE_WINDOW_SIZE):
        self.windows: Dict[str, SlidingWindowStats] = {
            feat: SlidingWindowStats(window_size) for feat in TRACKED_FEATURES
        }

    def update(self, feature_row: dict) -> None:
        for feat in TRACKED_FEATURES:
            if feat in feature_row and feature_row[feat] is not None:
                self.windows[feat].push(float(feature_row[feat]))

    def z_scores(self, feature_row: dict) -> Dict[str, float]:
        return {
            feat: self.windows[feat].z_score(float(feature_row[feat]))
            for feat in TRACKED_FEATURES
            if feat in feature_row and feature_row[feat] is not None
        }

    def is_ready(self) -> bool:
        """Baseline is only meaningful once at least one window has enough history."""
        return any(w.is_full() for w in self.windows.values())

    def snapshot(self) -> dict:
        return {feat: w.snapshot() for feat, w in self.windows.items()}


class BaselineEngine:
    """Manages UserBaseline objects for many users behind an LRU cache."""

    def __init__(self, cache_capacity: int = 512, window_size: int = BASELINE_WINDOW_SIZE):
        self._cache = LRUCache(capacity=cache_capacity)
        self.window_size = window_size

    def _get_or_create(self, user: str) -> UserBaseline:
        baseline = self._cache.get(user)
        if baseline is None:
            baseline = UserBaseline(window_size=self.window_size)
            self._cache.put(user, baseline)
        return baseline

    def process_message(self, user: str, feature_row: dict) -> Dict[str, float]:
        """
        Returns z-scores for this message *against the baseline built from
        prior messages* (i.e. update happens after scoring, so a message
        never gets scored against a window that includes itself).
        """
        baseline = self._get_or_create(user)
        z_scores = baseline.z_scores(feature_row) if baseline.is_ready() else {}
        baseline.update(feature_row)
        return z_scores

    def replay(self, feature_df: pd.DataFrame, user_col: str = "user") -> pd.DataFrame:
        """
        Process an entire (chronologically sorted) feature DataFrame,
        building baselines incrementally and returning z-score columns
        (prefixed z_) alongside the original features.
        """
        logger.info(f"Replaying {len(feature_df)} messages through baseline engine "
                    f"(window_size={self.window_size})...")
        z_rows = []
        for _, row in feature_df.iterrows():
            z = self.process_message(row[user_col], row.to_dict())
            z_rows.append({f"z_{k}": v for k, v in z.items()})
        z_df = pd.DataFrame(z_rows, index=feature_df.index)
        return pd.concat([feature_df, z_df], axis=1)


if __name__ == "__main__":
    engine = BaselineEngine()
    # toy example: user drifts from calm to urgent
    for i in range(35):
        row = {"sentiment_polarity": 0.1, "urgency_score": 0.0, "sentiment_subjectivity": 0.2,
               "readability_flesch": 60, "lexical_diversity": 0.7, "function_word_ratio": 0.4,
               "noun_ratio": 0.2, "verb_ratio": 0.15}
        z = engine.process_message("emp_test", row)
    # now an anomalous message
    anomalous = {"sentiment_polarity": -0.9, "urgency_score": 0.9, "sentiment_subjectivity": 0.9,
                 "readability_flesch": 20, "lexical_diversity": 0.3, "function_word_ratio": 0.1,
                 "noun_ratio": 0.5, "verb_ratio": 0.4}
    print(engine.process_message("emp_test", anomalous))
