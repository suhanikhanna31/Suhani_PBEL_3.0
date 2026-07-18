"""
PART 1 — Consent gating.

This is the "privacy-by-design" piece the mentor's notes flagged as the
differentiator: linguistic drift analysis on employee communications is
sensitive, so nothing downstream (features, baselines, drift scoring)
should run for a user without a recorded, current consent record.

Design: a simple CSV-backed consent ledger (data/interim/consent.csv) with
columns [user, consented, consented_on, scope]. In production this would
be backed by an actual HR/legal system of record — this module just
defines the *interface* the rest of the pipeline depends on, so swapping
the backing store later doesn't touch feature/model code.
"""
import logging
from pathlib import Path
from typing import Set

import pandas as pd

from src.config import DATA_INTERIM, CONSENT_REQUIRED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSENT_PATH = DATA_INTERIM / "consent.csv"


def ensure_consent_ledger(all_users: Set[str]) -> None:
    """
    Create a default consent ledger if one doesn't exist yet. Defaults
    everyone to consented=True ONLY for local dev/demo purposes — in a
    real deployment this file must come from an actual consent system,
    never be auto-generated as "opt-in by default."
    """
    if CONSENT_PATH.exists():
        return
    logger.warning(
        "No consent.csv found — generating a DEV-ONLY default ledger with "
        "all users opted in. Replace this with a real consent system of "
        "record before using on real employee data."
    )
    df = pd.DataFrame({
        "user": sorted(all_users),
        "consented": True,
        "consented_on": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "scope": "linguistic_drift_analysis_v1",
    })
    df.to_csv(CONSENT_PATH, index=False)


def get_consented_users() -> Set[str]:
    if not CONSENT_REQUIRED:
        return set()  # sentinel: caller should treat "consent not required" specially
    if not CONSENT_PATH.exists():
        return set()
    df = pd.read_csv(CONSENT_PATH)
    return set(df.loc[df["consented"] == True, "user"])  # noqa: E712


def filter_to_consented(df: pd.DataFrame, user_col: str = "user") -> pd.DataFrame:
    """Drop rows for any user without an active consent record. No-op if CONSENT_REQUIRED=False."""
    if not CONSENT_REQUIRED:
        return df

    ensure_consent_ledger(set(df[user_col].unique()))
    consented = get_consented_users()
    before = len(df)
    filtered = df[df[user_col].isin(consented)].copy()
    dropped = before - len(filtered)
    if dropped:
        logger.info(f"Consent gate: dropped {dropped}/{before} rows for non-consented users.")
    return filtered


def revoke_consent(user: str) -> None:
    """Mark a user as no longer consented. Downstream jobs should re-check before their next run."""
    if not CONSENT_PATH.exists():
        return
    df = pd.read_csv(CONSENT_PATH)
    df.loc[df["user"] == user, "consented"] = False
    df.to_csv(CONSENT_PATH, index=False)
    logger.info(f"Consent revoked for user={user}")
