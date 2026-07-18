"""
DSA: Aho-Corasick automaton for multi-pattern phrase matching.

Why: we scan every message against ~20-100+ urgency / social-engineering
phrases (config.URGENCY_PHRASES). Naively that's O(n_phrases * len(text))
per message via repeated substring search. Aho-Corasick builds a trie of
all patterns plus failure links, so a message of length m is scanned in
O(m + matches) *regardless of how many phrases we're looking for* — this
is what lets the phrase list grow to hundreds of entries without slowing
down per-message scanning.

Implementation: classic Aho-Corasick (trie + BFS-built failure links +
output links), built once at startup from config.URGENCY_PHRASES and
reused across all messages.
"""
from collections import deque
from typing import List, Dict, Tuple


class _Node:
    __slots__ = ("children", "fail", "output")

    def __init__(self):
        self.children: Dict[str, "_Node"] = {}
        self.fail: "_Node" = None
        self.output: List[str] = []  # phrases that end at this node


class AhoCorasick:
    def __init__(self, phrases: List[str]):
        self.root = _Node()
        self.root.fail = self.root
        self._phrases = [p.lower() for p in phrases]
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self):
        for phrase in self._phrases:
            node = self.root
            for ch in phrase:
                node = node.children.setdefault(ch, _Node())
            node.output.append(phrase)

    def _build_failure_links(self):
        queue = deque()
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for ch, child in current.children.items():
                queue.append(child)
                fail_node = current.fail
                while fail_node is not self.root and ch not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children.get(ch, self.root) if fail_node is not child else self.root
                if child.fail is child:
                    child.fail = self.root
                # merge output (a shorter phrase can be a suffix of a longer one)
                child.output += child.fail.output

    def scan(self, text: str) -> List[Tuple[str, int]]:
        """Return list of (phrase, end_index) matches in text. O(len(text) + matches)."""
        text = text.lower()
        node = self.root
        matches: List[Tuple[str, int]] = []

        for i, ch in enumerate(text):
            while node is not self.root and ch not in node.children:
                node = node.fail
            node = node.children.get(ch, self.root)
            for phrase in node.output:
                matches.append((phrase, i))
        return matches

    def urgency_score(self, text: str) -> float:
        """
        Simple normalized score: number of distinct matched phrases / total
        phrases in the lexicon, capped at 1.0. Used as one input feature to
        the risk model, not a verdict on its own.
        """
        matches = self.scan(text)
        if not matches or not self._phrases:
            return 0.0
        distinct = len({m[0] for m in matches})
        return min(distinct / max(len(self._phrases), 1) * 5, 1.0)  # scaled so 1-2 hits already register


if __name__ == "__main__":
    from src.config import URGENCY_PHRASES
    ac = AhoCorasick(URGENCY_PHRASES)
    sample = "Hey, this is urgent — please wire transfer the funds ASAP and don't tell anyone."
    print(ac.scan(sample))
    print("urgency_score:", ac.urgency_score(sample))
