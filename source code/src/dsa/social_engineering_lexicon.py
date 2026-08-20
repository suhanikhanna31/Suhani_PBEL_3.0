"""
DSA extension: categorized social-engineering lexicon.

Where this differs from config.URGENCY_PHRASES / trie_phrase_matcher.py's
single flat list: the phrases here are grouped by manipulation *technique*
(authority spoofing, artificial scarcity, isolation/secrecy framing, trust
exploitation, curiosity baiting) rather than lumped under one "urgency"
bucket. The grouping follows the standard attention-warfare / media-literacy
taxonomy of persuasion and manipulation techniques, not an ad hoc phrase
list — see docs/ATTENTION_WARFARE_LEXICON.md for the category rationale
and docs/ETHICS_AND_PRIVACY.md for why phrase-level matching stays
rule-based (auditable) rather than model-based here.

Each category reuses the existing AhoCorasick automaton unchanged, so a
message is still scanned in O(text length + matches) per category,
independent of how many phrases any single category holds. Running one
matcher per category (5 matchers here) rather than one matcher over the
union of all phrases is a deliberate trade-off: it costs a small constant
factor (5 scans instead of 1) in exchange for being able to report *which
category* fired, which is what makes the per-category breakdown usable as
an analyst-facing explanation rather than an opaque single number.
"""
from typing import Dict, List, Optional, Tuple

from src.dsa.trie_phrase_matcher import AhoCorasick
from src.config import SOCIAL_ENGINEERING_LEXICON


class SocialEngineeringLexicon:
    """
    Wraps one AhoCorasick matcher per manipulation category and exposes:
      - category_scores(): per-category urgency_score()-style values,
        useful for an analyst-facing "why was this flagged" breakdown
      - social_engineering_score(): a single aggregate feature, fed into
        the same drift-scoring pipeline as urgency_score

    The aggregate is a mean (not a max) of category scores, for the same
    reason drift_scoring.MAX_ABS_Z clips any single feature's z-score:
    one strongly-matched category shouldn't alone drown out the others.
    """

    def __init__(self, lexicon: Optional[Dict[str, List[str]]] = None):
        self._lexicon = lexicon or SOCIAL_ENGINEERING_LEXICON
        self._matchers: Dict[str, AhoCorasick] = {
            category: AhoCorasick(phrases)
            for category, phrases in self._lexicon.items()
        }

    def category_scores(self, text: str) -> Dict[str, float]:
        """One score per manipulation category, each in [0, 1]."""
        return {
            category: round(matcher.urgency_score(text), 4)
            for category, matcher in self._matchers.items()
        }

    def scan(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Raw (phrase, end_index) matches per category, for explanations."""
        return {
            category: matcher.scan(text)
            for category, matcher in self._matchers.items()
        }

    def social_engineering_score(self, text: str) -> float:
        """Single aggregate feature: mean of per-category scores."""
        scores = self.category_scores(text)
        if not scores:
            return 0.0
        return round(sum(scores.values()) / len(scores), 4)


if __name__ == "__main__":
    lexicon = SocialEngineeringLexicon()
    sample = (
        "This is a direct order from the CEO — keep this strictly between "
        "us and do not loop in your manager. Only available today, so "
        "please act as a valued colleague and handle it discreetly."
    )
    print("per-category:", lexicon.category_scores(sample))
    print("aggregate:", lexicon.social_engineering_score(sample))
