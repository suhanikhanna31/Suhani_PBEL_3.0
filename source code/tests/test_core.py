"""
Basic tests covering the DSA structures and core scoring logic. Run with:
    pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dsa.sliding_window import SlidingWindowStats
from src.dsa.trie_phrase_matcher import AhoCorasick
from src.dsa.top_k_heap import TopKRiskHeap
from src.dsa.lru_cache_baselines import LRUCache
from src.features.drift_scoring import score_drift
from src.data.anonymize import pseudonymize_user


class TestSlidingWindowStats:
    def test_mean_and_std(self):
        w = SlidingWindowStats(window_size=3)
        for v in [2, 4, 6]:
            w.push(v)
        assert w.mean == 4.0
        assert w.count == 3

    def test_eviction(self):
        w = SlidingWindowStats(window_size=2)
        w.push(1)
        w.push(2)
        w.push(3)  # should evict the 1
        assert w.count == 2
        assert w.mean == 2.5

    def test_zero_variance_zscore_is_zero(self):
        w = SlidingWindowStats(window_size=3)
        for _ in range(3):
            w.push(5.0)
        assert w.z_score(5.0) == 0.0

    def test_z_score_direction(self):
        w = SlidingWindowStats(window_size=5)
        for v in [1, 2, 3, 4, 5]:
            w.push(v)
        assert w.z_score(100) > 0
        assert w.z_score(-100) < 0


class TestAhoCorasick:
    def test_finds_known_phrase(self):
        ac = AhoCorasick(["urgent", "wire transfer"])
        matches = ac.scan("this is urgent, please wire transfer now")
        found_phrases = {m[0] for m in matches}
        assert "urgent" in found_phrases
        assert "wire transfer" in found_phrases

    def test_no_match_returns_empty(self):
        ac = AhoCorasick(["urgent"])
        assert ac.scan("hope you have a great weekend") == []

    def test_urgency_score_bounded(self):
        ac = AhoCorasick(["urgent", "asap", "wire transfer"])
        score = ac.urgency_score("urgent asap wire transfer urgent asap")
        assert 0.0 <= score <= 1.0


class TestTopKRiskHeap:
    def test_keeps_only_top_k(self):
        heap = TopKRiskHeap(k=2)
        for uid, score in [("a", 0.1), ("b", 0.9), ("c", 0.5)]:
            heap.push(uid, score)
        top = heap.top_k()
        assert len(top) == 2
        assert top[0].user_id == "b"
        assert top[1].user_id == "c"

    def test_sorted_descending(self):
        heap = TopKRiskHeap(k=5)
        for uid, score in [("a", 0.3), ("b", 0.9), ("c", 0.1)]:
            heap.push(uid, score)
        scores = [e.score for e in heap.top_k()]
        assert scores == sorted(scores, reverse=True)


class TestLRUCache:
    def test_basic_get_put(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_eviction_order(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")       # a is now most-recently-used
        cache.put("c", 3)    # should evict b
        assert "b" not in cache
        assert "a" in cache
        assert "c" in cache


class TestDriftScoring:
    def test_no_drift_when_no_zscores(self):
        result = score_drift({})
        assert result["drift_score"] == 0.0
        assert result["flagged_features"] == []

    def test_flags_above_threshold(self):
        result = score_drift({"z_urgency_score": 5.0, "z_sentiment_polarity": 0.1})
        assert "urgency_score" in result["flagged_features"]
        assert "sentiment_polarity" not in result["flagged_features"]

    def test_extreme_zscore_is_clipped(self):
        # a near-zero-variance feature shouldn't be able to blow up the score unboundedly
        result = score_drift({"z_urgency_score": 100000.0})
        assert result["drift_score"] <= 10.0 + 1e-6


class TestAnonymize:
    def test_deterministic(self):
        assert pseudonymize_user("alice") == pseudonymize_user("alice")

    def test_different_users_differ(self):
        assert pseudonymize_user("alice") != pseudonymize_user("bob")

    def test_format(self):
        assert pseudonymize_user("alice").startswith("emp_")
