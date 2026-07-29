"""
PART 2 — watsonx.ai foundation model client.

Real integration using the ibm-watsonx-ai SDK — not a stub. Point it at
your project by filling in .env (WATSONX_API_KEY, WATSONX_PROJECT_ID,
WATSONX_URL, WATSONX_MODEL_ID). config.WATSONX_ENABLED is True once both
the API key and project ID are present.

Role in this pipeline: NLP models (linguistic_features/stylometry) do the
heavy per-message scoring cheaply and locally. watsonx.ai is used
narrowly, on top of that, for the parts that benefit from a foundation
model's judgment rather than hand-built features:
  1. `explain_drift` — turns a flagged user's raw drift/z-score numbers
     into a plain-language rationale an analyst can read in one glance
     ("this is what watsonx.governance would call the 'model output' an
     audit log entry captures).
  2. `classify_message_risk` — a second opinion on a single message's
     risk category, used as a *complementary* signal to the local
     IsolationForest/RandomForest scores, not a replacement.

If credentials aren't configured, both functions degrade gracefully
(return None / a "watsonx not configured" note) so the rest of the
pipeline still runs without them — useful in the synthetic/dev setup.
"""
import logging
from typing import Optional

from src.config import (
    WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL,
    WATSONX_MODEL_ID, WATSONX_ENABLED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazily construct the watsonx ModelInference client (avoids import cost/errors when unused)."""
    global _model
    if _model is not None:
        return _model
    if not WATSONX_ENABLED:
        return None

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    
    # Overriding the stuck env config directly here to use the correct instruct model
    active_model_id = "ibm/granite-3-1-8b-instruct"

    _model = ModelInference(
        model_id=active_model_id,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={"decoding_method": "greedy", "max_new_tokens": 220, "temperature": 0.2},
    )
    return _model


def explain_drift(user_pseudonym: str, drift_summary: dict) -> Optional[str]:
    """
    drift_summary example:
      {"avg_drift_score": 3.1, "flagged_features": ["urgency_score", "sentiment_polarity"],
       "n_flagged_messages": 4, "n_messages": 87}
    Returns a short natural-language explanation, or None if watsonx isn't configured.
    """
    model = _get_model()
    if model is None:
        logger.info("watsonx.ai not configured (missing API key/project id) — skipping explanation.")
        return None

    prompt = (
        "You are assisting a security analyst reviewing an automated linguistic "
        "drift-detection alert. Do not name or assume identity — the user is "
        "pseudonymous. Given the statistics below, write a 2-3 sentence plain-"
        "language summary of what changed and why it was flagged, and note this "
        "is a statistical signal for human review, not a determination of wrongdoing.\n\n"
        f"Pseudonymous user ID: {user_pseudonym}\n"
        f"Average drift score: {drift_summary.get('avg_drift_score')}\n"
        f"Flagged features: {', '.join(drift_summary.get('flagged_features', [])) or 'none'}\n"
        f"Flagged messages: {drift_summary.get('n_flagged_messages')} of {drift_summary.get('n_messages')}\n\n"
        "Summary:"
    )
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        logger.error(f"watsonx.ai call failed: {e}")
        return None


def classify_message_risk(message_text: str) -> Optional[dict]:
    """
    Second-opinion classification for a single message. Returns
    {"category": ..., "rationale": ...} or None if unconfigured/failed.
    Categories are intentionally coarse — this augments, not replaces,
    the local supervised/unsupervised models.
    """
    model = _get_model()
    if model is None:
        return None

    prompt = (
        "Classify the following internal workplace message into exactly one "
        "category: 'routine', 'unusual_but_benign', or 'needs_review'. "
        "Respond in the format 'CATEGORY: <category> | REASON: <one sentence>'.\n\n"
        f"Message: {message_text}\n\nClassification:"
    )
    try:
        response = model.generate_text(prompt=prompt)
        text = response.strip() if isinstance(response, str) else str(response)
        category = "needs_review"
        for c in ("routine", "unusual_but_benign", "needs_review"):
            if c in text.lower():
                category = c
                break
        return {"category": category, "rationale": text}
    except Exception as e:
        logger.error(f"watsonx.ai call failed: {e}")
        return None


def answer_with_context(question: str, context_chunks: list) -> Optional[str]:
    """
    RAG grounding step for src/governance/rag.py. Given a question and a
    list of {"source": ..., "text": ...} chunks already retrieved by
    rag.retrieve() (TF-IDF, not this module's concern), ask watsonx.ai to
    answer using *only* that retrieved context — not its own general
    knowledge — and to say so plainly if the context doesn't actually
    answer the question, rather than guessing. Returns None if watsonx
    isn't configured, so callers can fall back to showing the raw
    retrieved passages instead (see rag._fallback_answer).
    """
    model = _get_model()
    if model is None:
        logger.info("watsonx.ai not configured (missing API key/project id) — skipping RAG generation.")
        return None

    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    prompt = (
        "You are answering a security analyst's question about how this "
        "insider-threat detection system itself works — its privacy design, "
        "architecture, or audit trail. Answer using ONLY the context passages "
        "below. If the passages don't actually contain the answer, say so "
        "plainly rather than guessing or using outside knowledge. Keep the "
        "answer to 2-4 sentences and cite which source(s) you drew from.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        logger.error(f"watsonx.ai call failed: {e}")
        return None


if __name__ == "__main__":
    print("WATSONX_ENABLED:", WATSONX_ENABLED)
    if WATSONX_ENABLED:
        print(explain_drift("emp_abc123", {
            "avg_drift_score": 3.4, "flagged_features": ["urgency_score", "sentiment_polarity"],
            "n_flagged_messages": 5, "n_messages": 92,
        }))
    else:
        print("Set WATSONX_API_KEY and WATSONX_PROJECT_ID in .env to test live calls.")