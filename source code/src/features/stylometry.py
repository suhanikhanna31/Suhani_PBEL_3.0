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

POS tagging uses spaCy's small English pipeline (`en_core_web_sm`) instead
of NLTK's averaged perceptron tagger — spaCy's tagger is a more accurate,
actively-maintained industrial NLP model (the standard choice on an IBM AI
team over NLTK's older statistical tagger) and, as a bonus, gives us a
single pipeline we could later extend with NER/dependency parsing for free.
Falls back to NLTK's tagger if the spaCy model isn't installed, so the rest
of the pipeline still runs without the extra download.
"""
import re
import logging
from collections import Counter

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_nlp = None  # None = not yet attempted, False = load failed, spaCy Language = ready


def _get_spacy_pipeline():
    """Lazily load spaCy's small English model, tagger only (no NER/lemmatizer needed here)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
        logger.info("Loaded spaCy pipeline: en_core_web_sm")
    except Exception as e:
        logger.warning(
            f"Could not load spaCy model 'en_core_web_sm' ({e}); "
            "falling back to NLTK's POS tagger. Install with: "
            "python -m spacy download en_core_web_sm"
        )
        _nlp = False
    return _nlp

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


_EMPTY_POS_RATIOS = {"noun_ratio": 0.0, "verb_ratio": 0.0, "adj_ratio": 0.0, "adv_ratio": 0.0}


def _nltk_pos_tags(text: str):
    """Fallback tagger, only imported/used if spaCy's model isn't installed."""
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        tokens = nltk.word_tokenize(text)
        return [t for _, t in nltk.pos_tag(tokens)]
    except Exception as e:
        logger.warning(f"NLTK POS tagging fallback also failed ({e}); returning no tags.")
        return []


def pos_ratio_features(text: str) -> dict:
    """
    Ratio of nouns/verbs/adjectives/adverbs to total tagged tokens.
    Tagged with spaCy's Penn Treebank-style fine-grained tags (`token.tag_`,
    e.g. NN/NNS, VB/VBZ, JJ, RB) so the NN*/VB*/JJ*/RB* prefix checks below
    behave the same as they did against NLTK's tagset. Falls back to NLTK if
    spaCy isn't available.
    """
    if not text.strip():
        return dict(_EMPTY_POS_RATIOS)

    nlp = _get_spacy_pipeline()
    if nlp:
        try:
            tags = [tok.tag_ for tok in nlp(text) if not tok.is_space]
        except Exception as e:
            logger.warning(f"spaCy POS tagging failed ({e}); falling back to NLTK.")
            tags = _nltk_pos_tags(text)
    else:
        tags = _nltk_pos_tags(text)

    if not tags:
        return dict(_EMPTY_POS_RATIOS)

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
