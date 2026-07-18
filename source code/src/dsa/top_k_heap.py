"""
DSA: bounded min-heap for maintaining the Top-K riskiest users.

Why: after scoring N users (N can be tens of thousands in CERT-scale data),
we usually only want the top K (e.g. 10-50) to show an analyst. Sorting
all N scores is O(N log N). A bounded min-heap of size K gives O(N log K)
instead — push every score, and whenever the heap exceeds size K, pop the
smallest. Since K << N, this is a meaningful win at scale and is the
standard "top-K of a stream" pattern.
"""
import heapq
from dataclasses import dataclass, field
from typing import List


@dataclass(order=True)
class RiskEntry:
    score: float
    user_id: str = field(compare=False)
    details: dict = field(compare=False, default_factory=dict)


class TopKRiskHeap:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self._heap: List[RiskEntry] = []

    def push(self, user_id: str, score: float, details: dict = None) -> None:
        entry = RiskEntry(score=score, user_id=user_id, details=details or {})
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0].score:
            heapq.heapreplace(self._heap, entry)
        # else: score too low to make the cut, O(1) discard

    def top_k(self) -> List[RiskEntry]:
        """Return entries sorted descending by risk score."""
        return sorted(self._heap, key=lambda e: e.score, reverse=True)


if __name__ == "__main__":
    heap = TopKRiskHeap(k=3)
    for uid, score in [("u1", 0.2), ("u2", 0.9), ("u3", 0.5), ("u4", 0.95), ("u5", 0.1)]:
        heap.push(uid, score)
    for e in heap.top_k():
        print(e.user_id, e.score)
