"""
PART 1 — Linguistic feature extraction.

Kept deliberately light per the project's own constraint: TextBlob/NLTK +
regex/DSA rather than heavy transformer-based NLP. Every function returns
plain floats so they drop straight into a feature DataFrame.

Features:
- sentiment_polarity, sentiment_subjectivity  (TextBlob)
- urgency_score                                (Aho-Corasick trie, DSA-optimized)
- readability (Flesch reading ease)            (textstat)
- lexical_diversity (type-token ratio)         (regex tokenization)
- avg_word_length, exclamation_ratio, caps_ratio, message_length
"""
import re
import logging
from functools import lru_cache

import pandas as pd
from textblob import TextBlob
import textstat

from src.config import URGENCY_PHRASES
from src.dsa.trie_phrase_matcher import AhoCorasick

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")

# Built once, reused for every message — this is the whole point of Aho-Corasick.
_urgency_matcher = AhoCorasick(URGENCY_PHRASES)


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


def sentiment_features(text: str) -> dict:
    blob = TextBlob(text)
    return {
        "sentiment_polarity": round(blob.sentiment.polarity, 4),      # -1 (neg) .. 1 (pos)
        "sentiment_subjectivity": round(blob.sentiment.subjectivity, 4),  # 0 (objective) .. 1 (subjective)
    }


def urgency_features(text: str) -> dict:
    return {"urgency_score": round(_urgency_matcher.urgency_score(text), 4)}


def readability_features(text: str) -> dict:
    try:
        score = textstat.flesch_reading_ease(text)
    except Exception:
        score = 0.0
    return {"readability_flesch": round(score, 2)}


def lexical_diversity_features(text: str) -> dict:
    tokens = _tokenize(text)
    if not tokens:
        return {"lexical_diversity": 0.0, "avg_word_length": 0.0}
    ttr = len(set(tokens)) / len(tokens)  # type-token ratio
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    return {"lexical_diversity": round(ttr, 4), "avg_word_length": round(avg_len, 2)}


def surface_features(text: str) -> dict:
    length = len(text)
    n_caps = sum(1 for c in text if c.isupper())
    n_exclaim = text.count("!")
    return {
        "message_length": length,
        "caps_ratio": round(n_caps / length, 4) if length else 0.0,
        "exclamation_ratio": round(n_exclaim / max(length, 1) * 100, 4),
    }


def extract_message_features(text: str) -> dict:
    """Full feature dict for a single message. This is the per-row unit of work."""
    text = text if isinstance(text, str) else ""
    feats = {}
    feats.update(sentiment_features(text))
    feats.update(urgency_features(text))
    feats.update(readability_features(text))
    feats.update(lexical_diversity_features(text))
    feats.update(surface_features(text))
    return feats


def extract_features_df(df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    """Vectorized-ish wrapper: applies extract_message_features across a DataFrame's content column."""
    logger.info(f"Extracting linguistic features for {len(df)} messages...")
    feat_rows = df[content_col].apply(extract_message_features)
    feat_df = pd.DataFrame(list(feat_rows), index=df.index)
    return pd.concat([df, feat_df], axis=1)


if __name__ == "__main__":
    sample = "This is urgent, please wire transfer the funds ASAP!!! Don't tell anyone."
    print(extract_message_features(sample))
