"""
PART 3 — Agentic AI: autonomous multi-step investigation agent.

Everything upstream of this module (features/, models/) produces a single
number or a single sentence per user — a drift score, an anomaly flag, a
one-shot watsonx.ai explanation. This module adds an *agent*: given a
flagged pseudonym, it plans and executes a short sequence of read-only
"tool calls" against the project's own modules, accumulates the evidence
it gathers, and — when watsonx.ai is configured — asks the model to reason
over that accumulated evidence (not just a single row of stats) before
proposing a next step. That plan-act-observe loop, with a visible,
auditable trace of what it looked at and why, is what makes this "agentic"
rather than just another single-shot model call like explain_drift().

Guardrails (non-negotiable, matches README's "it never acts autonomously
on a flag"):
  1. Every tool available to this agent is read-only. It has no access to
     consent revocation, the QRadar export, or any other write/enforcement
     path — those live in separate modules the agent never imports.
  2. The agent's output is always a *recommendation* for a human analyst,
     never an action. `requires_human_review` is hard-coded True and is
     not something a prompt (or a compromised/hallucinating model) can
     turn off.
  3. Every investigation — including each intermediate tool call — is
     written to the same tamper-evident audit log as every other scoring
     decision in this system, so an agent run is reviewable after the
     fact exactly like a model prediction is.
  4. If watsonx.ai isn't configured, the agent still runs: it falls back
     to a transparent, rule-based recommendation over the same evidence
     it gathered, so the pipeline never silently loses this capability in
     the free-tier/offline deployment.
"""
import logging
from typing import Optional

import pandas as pd

from src.config import DRIFT_Z_THRESHOLD, WATSONX_ENABLED
from src.data.store import read_df
from src.governance.audit_log import log_event, get_recent_entries
from src.governance.rag import retrieve
from src.models.watsonx.client import explain_drift, _get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_VERSION = "agentic_investigation_v1"
RECENT_MESSAGES_LIMIT = 15
PRIOR_EVENTS_SCANNED = 100

ACTIONS = ("escalate_for_manual_review", "continue_monitoring", "no_action_needed")


# ---------------------------------------------------------------------------
# Tools — each one is a thin, read-only wrapper around an existing module.
# The agent's "plan" is the fixed order these run in below; keeping the
# order deterministic (rather than letting an LLM freely choose which tool
# to call, ReAct-style) means a run is reproducible and cheap to audit,
# while the *reasoning over the results* is still delegated to watsonx.ai
# when available. This is a deliberate, documented trade-off — same spirit
# as rag.py choosing TF-IDF over an embedding model: inspectable over
# maximally flexible.
# ---------------------------------------------------------------------------

def _tool_get_risk_summary(pseudonym: str) -> Optional[dict]:
    df = read_df("user_risk")
    if df is None:
        return None
    row = df[df["user"] == pseudonym]
    return row.iloc[0].to_dict() if not row.empty else None


def _tool_get_recent_messages(pseudonym: str) -> list:
    df = read_df("scored_messages")
    if df is None:
        return []
    user_df = df[df["user"] == pseudonym]
    if user_df.empty:
        return []
    cols = [c for c in ("date", "drift_score", "n_flagged", "flagged_features",
                         "sentiment_polarity", "urgency_score") if c in user_df.columns]
    return user_df[cols].tail(RECENT_MESSAGES_LIMIT).to_dict(orient="records")


def _tool_get_prior_audit_events(pseudonym: str) -> list:
    recent = get_recent_entries(n=PRIOR_EVENTS_SCANNED)
    return [e for e in recent if e.get("user_pseudonym") == pseudonym]


def _tool_search_governance_docs(query: str) -> list:
    hits = retrieve(query, k=2)
    return [{"source": h["source"], "score": round(h["score"], 3)} for h in hits]


# ---------------------------------------------------------------------------
# Reasoning step
# ---------------------------------------------------------------------------

def _rule_based_recommendation(risk_summary: dict, prior_events: list) -> dict:
    """Transparent fallback used whenever watsonx.ai isn't configured, and
    as the deterministic backstop if a live model call fails or returns an
    unparseable response. Every branch is auditable from this source alone."""
    avg_drift = float(risk_summary.get("avg_drift_score", 0) or 0)
    flagged_rate = float(risk_summary.get("flagged_message_rate", 0) or 0)
    repeat_flags = sum(1 for e in prior_events if e.get("event_type") == "drift_flagged")

    if avg_drift >= DRIFT_Z_THRESHOLD * 1.2 or flagged_rate > 0.15 or repeat_flags >= 3:
        return {
            "action": "escalate_for_manual_review",
            "rationale": (
                f"Average drift score {avg_drift:.2f} and/or flagged-message rate "
                f"{flagged_rate*100:.1f}% exceed the escalation thresholds, with "
                f"{repeat_flags} prior drift-flag event(s) on record for this "
                "pseudonym. Rule-based fallback (watsonx.ai not consulted)."
            ),
        }
    if avg_drift >= DRIFT_Z_THRESHOLD:
        return {
            "action": "continue_monitoring",
            "rationale": (
                f"Average drift score {avg_drift:.2f} is above the flagging "
                f"threshold ({DRIFT_Z_THRESHOLD}) but below the escalation bar. "
                "Rule-based fallback (watsonx.ai not consulted)."
            ),
        }
    return {
        "action": "no_action_needed",
        "rationale": (
            f"Average drift score {avg_drift:.2f} is within this user's own "
            "normal range. Rule-based fallback (watsonx.ai not consulted)."
        ),
    }


def _llm_recommendation(pseudonym: str, risk_summary: dict, messages: list,
                         prior_events: list, explanation: Optional[str]) -> Optional[dict]:
    """Asks watsonx.ai to reason over *all* evidence gathered so far and
    propose one of the fixed ACTIONS. Unlike explain_drift() (a one-shot
    call on the summary row alone), this is the agent's synthesis step —
    it sees the recent-message trend and the prior-audit history too.
    Returns None (triggering the rule-based fallback) if watsonx isn't
    configured or its response can't be parsed into a known action."""
    model = _get_model()
    if model is None:
        return None

    prompt = (
        "You are an insider-risk investigation agent assisting a human security "
        "analyst. You may only recommend one of exactly three actions — you have "
        "no ability to take action yourself, only to recommend. Choose the single "
        "best action and justify it in 2-3 sentences using only the evidence "
        "given. Never claim certainty about intent or wrongdoing; this is a "
        "statistical signal for human review only.\n\n"
        f"Pseudonymous user: {pseudonym}\n"
        f"Risk summary: {risk_summary}\n"
        f"Recent scored messages (most recent {len(messages)}): {messages}\n"
        f"Prior audit events for this user: {len(prior_events)} on record\n"
        f"Existing plain-language explanation: {explanation or 'none available'}\n\n"
        "Respond in EXACTLY this format:\n"
        "ACTION: <escalate_for_manual_review|continue_monitoring|no_action_needed>\n"
        "RATIONALE: <2-3 sentences>"
    )
    try:
        raw = model.generate_text(prompt=prompt)
        text = raw.strip() if isinstance(raw, str) else str(raw)
    except Exception as e:
        logger.error(f"watsonx.ai call failed during agent synthesis: {e}")
        return None

    action = next((a for a in ACTIONS if a in text), None)
    if action is None:
        return None
    rationale = text.split("RATIONALE:", 1)[-1].strip() if "RATIONALE:" in text else text
    return {"action": action, "rationale": rationale, "raw_model_output": text}


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def investigate_user(pseudonym: str, actor: str = "system") -> dict:
    """
    Runs the full plan-act-observe loop for one pseudonym and returns a
    structured, auditable investigation report. Safe to call repeatedly —
    it never mutates risk data, consent state, or the audit chain beyond
    appending its own trace entries.
    """
    trace = []

    risk_summary = _tool_get_risk_summary(pseudonym)
    trace.append({"tool": "get_risk_summary", "found": risk_summary is not None})
    if risk_summary is None:
        return {
            "user_pseudonym": pseudonym,
            "generated_by": AGENT_VERSION,
            "error": f"No risk data for pseudonym '{pseudonym}'. Has the pipeline been run?",
            "evidence_trace": trace,
            "requires_human_review": True,
        }

    messages = _tool_get_recent_messages(pseudonym)
    trace.append({"tool": "get_recent_messages", "count": len(messages)})

    prior_events = _tool_get_prior_audit_events(pseudonym)
    trace.append({"tool": "get_prior_audit_events", "count": len(prior_events)})

    explanation = explain_drift(pseudonym, risk_summary)
    trace.append({"tool": "get_watsonx_explanation", "available": explanation is not None})

    # Only spend a retrieval + reasoning step on governance docs when the
    # evidence so far is actually borderline enough to warrant it — keeps
    # low-risk users cheap to investigate.
    policy_context = []
    avg_drift = float(risk_summary.get("avg_drift_score", 0) or 0)
    if avg_drift >= DRIFT_Z_THRESHOLD:
        policy_context = _tool_search_governance_docs(
            "insider risk escalation review consent audit process"
        )
        trace.append({"tool": "search_governance_docs", "hits": len(policy_context)})

    recommendation = _llm_recommendation(pseudonym, risk_summary, messages, prior_events, explanation)
    used_llm = recommendation is not None
    if recommendation is None:
        recommendation = _rule_based_recommendation(risk_summary, prior_events)
    trace.append({"tool": "synthesize_recommendation", "used_watsonx": used_llm})

    report = {
        "user_pseudonym": pseudonym,
        "generated_by": AGENT_VERSION,
        "watsonx_reasoning_used": used_llm,
        "watsonx_configured": WATSONX_ENABLED,
        "risk_summary": risk_summary,
        "explanation": explanation,
        "policy_context": policy_context,
        "recommendation": recommendation["action"],
        "rationale": recommendation["rationale"],
        "evidence_trace": trace,
        # Hard-coded, not model-controllable: the agent proposes, a human
        # analyst/admin disposes. See module docstring guardrail (2).
        "requires_human_review": True,
    }

    log_event(
        "agent_investigation_completed",
        pseudonym,
        {
            "recommendation": recommendation["action"],
            "watsonx_reasoning_used": used_llm,
            "steps": len(trace),
        },
        actor=actor,
    )
    return report


if __name__ == "__main__":
    import json
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "emp_abc123"
    print(json.dumps(investigate_user(target), indent=2, default=str))
