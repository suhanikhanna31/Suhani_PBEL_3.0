"""
Retrieval-augmented question answering over this project's own governance
documentation and audit trail.

Role in this project: watsonx Assistant (src/api/routes/assistant.py)
already lets an analyst ask structured questions like "show me the top
risky users" by calling back into ranked risk data. This module adds a
second, complementary capability — open-ended questions about *how the
system behaves and why*, such as "what happens if a user revokes
consent?" or "was anything unusual logged recently?" — answered by
retrieving from the project's real ETHICS_AND_PRIVACY.md/ARCHITECTURE.md
docs and recent real audit_log.py entries, then (optionally) asking
watsonx.ai to phrase a grounded answer from exactly that retrieved
context rather than from the model's general training.

Deliberately NOT using an embedding model or vector database. This
project already treats "well-audited, explainable tools over the newest
available library" as a stated design principle (see README's Python
toolchain table), and a second heavy model is exactly what the free-tier
deploy's 512MB RAM ceiling can't absorb (see README's "Deployment
status" — transformers/torch were already stripped from that deploy for
this same reason). TF-IDF + cosine similarity, both already available via
scikit-learn (an existing dependency, no new package), is a transparent,
inspectable retrieval method: a reviewer can see exactly which words
matched and why a chunk was retrieved, which matters for the same reason
Section 6.2 of the report keeps urgency/readability scoring rule-based
even after adopting a transformer for sentiment.

Does not touch drift scoring, the classifiers, or any file this module
doesn't own — same pattern as bias_audit.py and openscale_monitor.py,
called independently rather than wired into the scoring path.
"""
import logging
import re
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ROOT_DIR
from src.governance.audit_log import get_recent_entries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = ROOT_DIR / "docs"
DOC_FILES = ["ETHICS_AND_PRIVACY.md", "ARCHITECTURE.md"]

MIN_CHUNK_CHARS = 40  # drops stray headers/blank fragments, keeps real paragraphs


def _chunk_markdown(text: str, source: str) -> list:
    """Split a markdown file into chunks along ## / # headers, so each chunk
    is one coherent section rather than an arbitrary character window."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) >= MIN_CHUNK_CHARS:
            chunks.append({"source": source, "text": section})
    return chunks


def _load_doc_chunks() -> list:
    chunks = []
    for filename in DOC_FILES:
        path = DOCS_DIR / filename
        if not path.exists():
            logger.warning(f"RAG corpus: {path} not found, skipping.")
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(_chunk_markdown(text, source=filename))
    return chunks


def _format_audit_entry(entry: dict) -> str:
    details = ", ".join(f"{k}={v}" for k, v in entry.get("details", {}).items())
    user = entry.get("user_pseudonym") or "no user (system-level event)"
    return (
        f"Audit log entry at {entry.get('timestamp')}: event '{entry.get('event_type')}' "
        f"recorded by actor '{entry.get('actor')}' for {user}. Details: {details or 'none'}."
    )


def _load_audit_chunks(n: int = 50) -> list:
    entries = get_recent_entries(n=n)
    return [{"source": "audit_log", "text": _format_audit_entry(e)} for e in entries]


def build_corpus(include_audit_log: bool = True, audit_n: int = 50) -> list:
    """
    Rebuilt on every call rather than cached: the audit log grows during a
    running process, and the doc files are small enough (a few KB) that
    re-reading them every call costs nothing worth optimizing for — matching
    this project's general preference for correctness over premature caching
    (see LRU cache section of the report for where bounded caching *is*
    actually justified, vs. here where it isn't).
    """
    corpus = _load_doc_chunks()
    if include_audit_log:
        corpus.extend(_load_audit_chunks(n=audit_n))
    return corpus


def retrieve(question: str, k: int = 4, corpus: Optional[list] = None) -> list:
    """
    Return the top-k chunks most relevant to `question`, each with a
    similarity score, ranked highest first. Returns [] if the corpus is
    empty (e.g. no docs/ directory in this deployment) rather than raising.
    """
    if corpus is None:
        corpus = build_corpus()
    if not corpus:
        logger.warning("RAG retrieve() called with an empty corpus.")
        return []

    texts = [c["text"] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(query_vector, doc_vectors)[0]
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in ranked[:k]:
        if score <= 0:
            continue  # no lexical overlap at all — don't pad results with noise
        results.append({**chunk, "score": float(score)})
    return results


def answer_question(question: str, k: int = 4) -> dict:
    """
    Full RAG flow: retrieve real context, then (if watsonx.ai is configured)
    ask it to phrase a grounded answer from exactly that context. If watsonx
    isn't configured, return the retrieved chunks directly and say so
    plainly — the same "never silently fabricate, state the stub plainly"
    rule explain_drift() already follows in src/models/watsonx/client.py.
    """
    from src.models.watsonx.client import answer_with_context
    from src.governance.audit_log import log_event

    chunks = retrieve(question, k=k)

    if not chunks:
        result = {
            "answer": (
                "No relevant passage found in the project's governance docs or "
                "recent audit log for this question. Try rephrasing, or ask "
                "something closer to consent, anonymization, audit logging, "
                "or pipeline architecture."
            ),
            "sources": [],
            "watsonx_generated": False,
        }
    else:
        generated = answer_with_context(question, chunks)
        result = {
            "answer": generated if generated else _fallback_answer(chunks),
            "sources": [{"source": c["source"], "score": round(c["score"], 3)} for c in chunks],
            "watsonx_generated": generated is not None,
        }

    log_event(
        "rag_query",
        user_pseudonym=None,  # asked by an analyst about the system itself, not about a user
        details={
            "question": question,
            "n_sources_retrieved": len(chunks),
            "watsonx_generated": result["watsonx_generated"],
        },
    )
    return result


def _fallback_answer(chunks: list) -> str:
    """No model configured — return the retrieved passages themselves,
    clearly labeled, rather than nothing."""
    lines = ["watsonx.ai not configured — showing retrieved passages directly:\n"]
    for c in chunks:
        lines.append(f"[{c['source']}, relevance {c['score']:.2f}]\n{c['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    for q in [
        "What happens if a user revokes consent?",
        "How does the audit log detect tampering?",
        "Was anything unusual logged recently?",
    ]:
        print(f"\n=== {q} ===")
        result = answer_question(q)
        print(result["answer"])
        print("sources:", result["sources"])