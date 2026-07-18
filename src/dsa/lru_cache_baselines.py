"""
DSA: LRU cache for per-user baseline objects.

Why: drift_scoring.py needs each user's SlidingWindowStats baseline on
every scored message. Rebuilding a baseline from scratch (rescanning that
user's history) on every lookup is expensive; keeping *all* users'
baselines resident in memory forever doesn't scale either. An LRU cache
gives O(1) get/put and bounds memory to `capacity` most-recently-active
users, which matches real traffic patterns (most activity comes from a
small recently-active subset of employees at any given time).

Implemented with an OrderedDict for O(1) average get/put + O(1) move-to-end.
"""
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    def __init__(self, capacity: int = 512):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)  # mark as most-recently-used
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict least-recently-used

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)


if __name__ == "__main__":
    cache = LRUCache(capacity=2)
    cache.put("alice", "baseline_a")
    cache.put("bob", "baseline_b")
    cache.get("alice")           # alice now most-recently-used
    cache.put("carol", "baseline_c")  # evicts bob (least-recently-used)
    print("bob" in cache, "alice" in cache, "carol" in cache)
