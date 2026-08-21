"""
DSA: hand-rolled communication-graph structure for social-topology drift.

Every other module in this project asks "how does *what* someone writes
compare to their own history" (sentiment, urgency, stylometry — see
baseline_engine.py). This module asks a different, structurally distinct
question: "who is this person talking to, and has that circle changed
shape?" Social withdrawal — a shrinking, more concentrated communication
circle — is one of the most consistently documented pre-incident behaviors
in real counterintelligence casework, and it is invisible to any purely
linguistic drift model, no matter how good the language features are.

Why a plain dict-of-dicts instead of networkx:
A full graph library is overkill for what this needs (per-user degree and
concentration over a bounded recent window) and it's an extra dependency
this project deliberately avoids (see requirements.txt — every dependency
here is already load-bearing elsewhere). A dict-of-dicts adjacency list —
`graph[sender][recipient] = message_count` — gives O(1) average-case edge
lookup/update, same complexity class as an adjacency-list graph in any
textbook, with zero new dependencies. This is the same "hand-rolled DSA
over an off-the-shelf library" choice already made for Aho-Corasick
(trie_phrase_matcher.py) over a regex/library alternative.

Two structures are kept side by side, deliberately:
  - `_adjacency`   : an unbounded, all-time edge-count graph — the
                     lifetime social graph, useful for audit/debugging.
  - `_recent_contacts` : a *bounded* deque per sender (maxlen=window_size),
                     the sliding window this module's actual features are
                     computed from — the same BASELINE_WINDOW_SIZE sliding-
                     window philosophy as sliding_window.py, applied to
                     "who," not "how."

Two derived features, both O(window_size) to compute (bounded, not
proportional to total history):
  - degree(user)        : count of distinct people talked to in the
                           window. A shrinking number over time is the
                           withdrawal signal.
  - concentration(user) : a Herfindahl-Hirschman-style index
                           (sum of squared contact shares) over the window.
                           0 → messages spread evenly across many contacts;
                           1 → every message in the window went to one
                           person. Rising concentration alongside falling
                           degree is the sharpest version of the withdrawal
                           pattern — talking to fewer people, and almost
                           all of it to one of them.
"""
from collections import deque, Counter
from typing import Dict, Iterable, List

from src.config import BASELINE_WINDOW_SIZE


class CommunicationGraph:
    def __init__(self, window_size: int = BASELINE_WINDOW_SIZE):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        # dict-of-dicts adjacency: sender -> {recipient -> lifetime count}
        self._adjacency: Dict[str, Dict[str, int]] = {}
        # bounded sliding window of the most recent contacts per sender
        self._recent_contacts: Dict[str, deque] = {}

    def record_message(self, sender: str, recipients: Iterable[str]) -> None:
        """
        Registers one message's edges. O(r) where r = number of recipients
        on this message (almost always small — a handful of To/Cc/Bcc
        addresses), not proportional to the graph's total size.
        """
        recipients = [r for r in recipients if r]
        if not recipients:
            return

        adj_row = self._adjacency.setdefault(sender, {})
        for r in recipients:
            adj_row[r] = adj_row.get(r, 0) + 1

        window = self._recent_contacts.setdefault(sender, deque(maxlen=self.window_size))
        window.extend(recipients)

    def degree(self, user: str) -> int:
        """Distinct contacts within the sliding window. O(window_size)."""
        window = self._recent_contacts.get(user)
        if not window:
            return 0
        return len(set(window))

    def concentration(self, user: str) -> float:
        """
        Herfindahl-style concentration index over the sliding window:
        sum((contact_i_count / total_count)^2), range [1/window_size, 1].
        Low = evenly spread across many contacts. High = a shrinking,
        more concentrated circle — the withdrawal signal this module
        exists to catch.
        """
        window = self._recent_contacts.get(user)
        if not window:
            return 0.0
        counts = Counter(window)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return sum((c / total) ** 2 for c in counts.values())

    def lifetime_degree(self, user: str) -> int:
        """Distinct contacts across the user's entire history (unbounded), for audit/debug only."""
        return len(self._adjacency.get(user, {}))

    def network_features(self, user: str) -> dict:
        """The two features that feed the baseline/drift pipeline."""
        return {
            "contact_degree": float(self.degree(user)),
            "interaction_concentration": float(self.concentration(user)),
        }

    def top_contacts(self, user: str, k: int = 5) -> List[tuple]:
        """Most-messaged contacts within the current window, most recent-heavy first — for analyst drill-down, not scoring."""
        window = self._recent_contacts.get(user)
        if not window:
            return []
        return Counter(window).most_common(k)


if __name__ == "__main__":
    # self-test: a user's circle narrows from 5 regular contacts down to 1
    g = CommunicationGraph(window_size=10)
    wide_circle = ["a@co.com", "b@co.com", "c@co.com", "d@co.com", "e@co.com"]
    for _ in range(10):
        g.record_message("emp_test", [wide_circle[_ % len(wide_circle)]])
    print("wide circle:", g.network_features("emp_test"))

    for _ in range(10):
        g.record_message("emp_test", ["single_contact@external.com"])
    print("narrowed circle:", g.network_features("emp_test"))
