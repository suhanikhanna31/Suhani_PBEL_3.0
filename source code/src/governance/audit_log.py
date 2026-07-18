"""
PART 2 — Audit logging (watsonx.governance-style).

Every scoring decision, drift flag, and watsonx.ai call that touches a
real (pseudonymized) user should leave an append-only, tamper-evident
trail — this is the "responsibly audit/limit this" requirement from the
mentor's notes made concrete.

Design: JSON Lines file, each entry hash-chained to the previous one
(entry.prev_hash = hash of previous entry), so any retroactive edit or
deletion breaks the chain and is detectable by verify_chain(). This is
the same core idea as a blockchain's tamper-evidence, applied to a single
append-only log rather than a distributed ledger — appropriate here since
we don't need consensus, just detectability.

In a real IBM Cloud deployment this would write to a governed store
(e.g. watsonx.governance's own audit trail, or an immutable Cloud Object
Storage bucket with retention lock) instead of a local file — swap
_append() to do that without touching call sites.
"""
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import AUDIT_LOG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _hash_entry(entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _last_hash() -> str:
    if not AUDIT_LOG_PATH.exists() or AUDIT_LOG_PATH.stat().st_size == 0:
        return "0" * 64  # genesis
    with open(AUDIT_LOG_PATH, "rb") as f:
        last_line = None
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        return "0" * 64
    return json.loads(last_line)["entry_hash"]


def log_event(event_type: str, user_pseudonym: Optional[str], details: dict,
              actor: str = "system") -> dict:
    """
    event_type examples: 'drift_flagged', 'watsonx_explanation_generated',
    'model_prediction', 'consent_revoked', 'api_access'
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "user_pseudonym": user_pseudonym,  # NEVER a real username — pseudonyms only
        "actor": actor,
        "details": details,
        "prev_hash": _last_hash(),
    }
    entry["entry_hash"] = _hash_entry(entry)

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return entry


def verify_chain() -> bool:
    """Walk the whole log and confirm each entry's prev_hash matches the prior entry's hash."""
    if not AUDIT_LOG_PATH.exists():
        return True

    expected_prev = "0" * 64
    with open(AUDIT_LOG_PATH, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["prev_hash"] != expected_prev:
                logger.error(f"Audit chain broken at line {i}: prev_hash mismatch.")
                return False
            claimed_hash = entry.pop("entry_hash")
            recomputed = _hash_entry(entry)
            if claimed_hash != recomputed:
                logger.error(f"Audit chain broken at line {i}: entry_hash mismatch (tampered content).")
                return False
            expected_prev = claimed_hash
    logger.info("Audit chain verified OK.")
    return True


if __name__ == "__main__":
    log_event("drift_flagged", "emp_abc123", {"drift_score": 3.2, "flagged_features": ["urgency_score"]})
    log_event("model_prediction", "emp_abc123", {"model": "random_forest", "risk_proba": 0.81})
    print("Chain valid:", verify_chain())
