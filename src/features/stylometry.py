"""
PART 1 — Stylometry features.

Stylometry captures *how* someone writes, independent of *what* they're
writing about — function-word frequencies and punctuation habits are
known in authorship-attribution literature to be fairly stable per-person
and to shift when someone is impersonated, stressed, or writing under
different constraints (e.g. a compromised account being used by someone
else, or a genuine insider's writing style changing under stress). That's
why this sits alongside sentiment/urgency as a *separate* signal for
baseline-vs-current drift, rather than folded into linguistic_features.py.

Requires NLTK's punkt + averaged_perceptron_tagger for POS tagging
(downloaded on first import if missing).
"""
import re
import logging
from collections import Counter

import nltk
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # some pkg names only apply to newer/older nltk; safe to skip if unavailable

# A compact function-word list (pronouns, prepositions, conjunctions, articles) —
# these are the words stylometry research leans on because topic barely affects them.
FUNCTION_WORDS = set("""
i me my we our you your he him his she her it its they them their
the a an this that these those
and or but if because as until while
of at by for with about against between into through during
in on to from up down out off over under
is am are was were be been being have has had do does did
not no nor
""".split())

_WORD_RE = re.compile(r"[A-Za-z']+")


def function_word_features(text: str) -> dict:
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return {"function_word_ratio": 0.0}
    fw_count = sum(1 for t in tokens if t in FUNCTION_WORDS)
    return {"function_word_ratio": round(fw_count / len(tokens), 4)}


def punctuation_features(text: str) -> dict:
    length = max(len(text), 1)
    counts = Counter(text)
    return {
        "comma_ratio": round(counts.get(",", 0) / length, 4),
        "period_ratio": round(counts.get(".", 0) / length, 4),
        "question_ratio": round(counts.get("?", 0) / length, 4),
        "ellipsis_count": text.count("..."),
    }


def pos_ratio_features(text: str) -> dict:
    """Ratio of nouns/verbs/adjectives/adverbs to total tagged tokens."""
    try:
        tokens = nltk.word_tokenize(text)
        tags = [t for _, t in nltk.pos_tag(tokens)]
    except Exception:
        return {"noun_ratio": 0.0, "verb_ratio": 0.0, "adj_ratio": 0.0, "adv_ratio": 0.0}

    if not tags:
        return {"noun_ratio": 0.0, "verb_ratio": 0.0, "adj_ratio": 0.0, "adv_ratio": 0.0}

    n = len(tags)
    noun = sum(1 for t in tags if t.startswith("NN")) / n
    verb = sum(1 for t in tags if t.startswith("VB")) / n
    adj = sum(1 for t in tags if t.startswith("JJ")) / n
    adv = sum(1 for t in tags if t.startswith("RB")) / n
    return {
        "noun_ratio": round(noun, 4),
        "verb_ratio": round(verb, 4),
        "adj_ratio": round(adj, 4),
        "adv_ratio": round(adv, 4),
    }


def extract_stylometry_features(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    feats = {}
    feats.update(function_word_features(text))
    feats.update(punctuation_features(text))
    feats.update(pos_ratio_features(text))
    return feats


def extract_stylometry_df(df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    logger.info(f"Extracting stylometry features for {len(df)} messages...")
    feat_rows = df[content_col].apply(extract_stylometry_features)
    feat_df = pd.DataFrame(list(feat_rows), index=df.index)
    return pd.concat([df, feat_df], axis=1)


if __name__ == "__main__":
    print(extract_stylometry_features("Hi there, could you please send me the file? Thanks!"))
