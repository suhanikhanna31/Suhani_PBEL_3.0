"""
DSA: fixed-size sliding window with O(1) amortized rolling mean/variance.

Why not just call np.mean()/np.std() on a slice every time?
Recomputing mean/variance from scratch on every new message is O(w) per
update where w = window size. Across thousands of users each producing
hundreds of messages, that's O(n * w) total. This structure updates
incrementally in O(1) per insertion (amortized), giving O(n) total —
the difference matters once this runs per-user, per-day, at CERT-dataset
scale (tens of thousands of employees).

Approach: a deque holds the last `w` values. On insert we update a running
sum and running sum-of-squares in O(1); when the window overflows we evict
the oldest value and adjust the same running totals in O(1). Mean and
(population) variance are then simple arithmetic, no rescans.

This is a simplified two-pass-free variant of Welford's algorithm adapted
for a *bounded* window (Welford's classic form is for an unbounded stream).
"""
from collections import deque


class SlidingWindowStats:
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._values = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0

    def push(self, value: float) -> None:
        """Add a new value, evicting the oldest if the window is full. O(1) amortized."""
        if len(self._values) == self.window_size:
            oldest = self._values[0]  # will be auto-evicted by deque's maxlen
            self._sum -= oldest
            self._sum_sq -= oldest * oldest
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return self._sum / len(self._values)

    @property
    def variance(self) -> float:
        """Population variance, clamped to >= 0 to guard against fp drift."""
        n = len(self._values)
        if n == 0:
            return 0.0
        mean = self.mean
        var = (self._sum_sq / n) - (mean * mean)
        return max(var, 0.0)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def z_score(self, value: float) -> float:
        """How many std-devs `value` is from this window's mean. 0.0 std -> 0.0 (no drift signal yet)."""
        std = self.std
        if std == 0.0:
            return 0.0
        return (value - self.mean) / std

    def is_full(self) -> bool:
        return len(self._values) == self.window_size

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "window_size": self.window_size,
        }


if __name__ == "__main__":
    # quick self-test
    w = SlidingWindowStats(window_size=5)
    for v in [1, 2, 3, 4, 5, 100]:  # 100 should push out the "1" and spike the mean
        w.push(v)
    print(w.snapshot())
    print("z-score of 100 given current window:", w.z_score(100))
