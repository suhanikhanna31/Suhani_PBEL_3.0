"""
PART 1 — Linguistic feature extraction.

Sentiment now comes from a fine-tuned transformer (DistilBERT/SST-2) rather
than a lexicon-based scorer like TextBlob — the kind of model-driven NLP
that's a better showcase for an IBM AI internship (Hugging Face
`transformers`, same family of tooling watsonx.ai itself is built on)
instead of a hand-built word-polarity list. Everything downstream still
just sees plain floats, so the feature DataFrame / drift-scoring / baseline
code doesn't need to change at all — only *how* the numbers are produced.

Features:
- sentiment_polarity, sentiment_subjectivity   (DistilBERT SST-2 transformer,
                                                 TextBlob fallback if the model
                                                 can't be loaded)
- urgency_score                                (Aho-Corasick trie, DSA-optimized)
- readability (Flesch reading ease)            (textstat)
- lexical_diversity (type-token ratio)         (regex tokenization)
- avg_word_length, exclamation_ratio, caps_ratio, message_length
"""
import re
import logging

import pandas as pd
import textstat

from src.config import URGENCY_PHRASES
from src.dsa.trie_phrase_matcher import AhoCorasick

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")

# Built once, reused for every message — this is the whole point of Aho-Corasick.
_urgency_matcher = AhoCorasick(URGENCY_PHRASES)

_SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_sentiment_pipeline = None  # None = not yet attempted, False = load failed, pipeline = ready


def _get_sentiment_pipeline():
    """
    Lazily load the Hugging Face sentiment pipeline (mirrors the lazy-loading
    pattern already used for the watsonx client) so importing this module
    doesn't force a model download/import cost for callers who don't need it,
    and so the whole pipeline still runs (with a fallback) in an offline/CI
    environment that can't fetch model weights.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=_SENTIMENT_MODEL_NAME,
            tokenizer=_SENTIMENT_MODEL_NAME,
            top_k=None,  # return scores for both POSITIVE and NEGATIVE
        )
        logger.info(f"Loaded transformer sentiment model: {_SENTIMENT_MODEL_NAME}")
    except Exception as e:
        logger.warning(
            f"Could not load transformer sentiment model ({e}); "
            "falling back to TextBlob for sentiment scoring."
        )
        _sentiment_pipeline = False
    return _sentiment_pipeline


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


def sentiment_features(text: str) -> dict:
    """
    Transformer-based sentiment via DistilBERT fine-tuned on SST-2.

    Keeps the same output schema/semantics as before so nothing downstream
    (baseline engine, drift scoring, anomaly detection, API) needs updating:
      - sentiment_polarity: -1 (negative) .. 1 (positive), now derived as
        P(positive) - P(negative) from the model's softmax output.
      - sentiment_subjectivity: 0 .. 1, now repurposed as a "how ambiguous/
        mixed is this message" signal (1 - model confidence) — a message the
        model is unsure about (~50/50 pos vs neg) behaves like TextBlob's
        "subjective" text, while a confident call behaves like "objective".
    """
    text = text.strip()
    if not text:
        return {"sentiment_polarity": 0.0, "sentiment_subjectivity": 0.0}

    clf = _get_sentiment_pipeline()
    if clf:
        try:
            scores = {r["label"].upper(): r["score"] for r in clf(text[:512])[0]}
            pos = scores.get("POSITIVE", 0.0)
            neg = scores.get("NEGATIVE", 0.0)
            polarity = pos - neg
            confidence = max(pos, neg)
            return {
                "sentiment_polarity": round(polarity, 4),
                "sentiment_subjectivity": round(1 - confidence, 4),
            }
        except Exception as e:
            logger.warning(f"Transformer sentiment inference failed ({e}); falling back to TextBlob.")

    from textblob import TextBlob
    blob = TextBlob(text)
    return {
        "sentiment_polarity": round(blob.sentiment.polarity, 4),
        "sentiment_subjectivity": round(blob.sentiment.subjectivity, 4),
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
