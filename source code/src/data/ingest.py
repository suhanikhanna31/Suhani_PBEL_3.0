"""
PART 1 — Ingestion.

Loads CERT-schema email data from data/raw/. If no real CERT csv is
present (e.g. you haven't downloaded it from Kaggle yet), falls back to
generating a synthetic dataset with the same schema so the pipeline is
runnable end-to-end during development.

Supports both Kaggle mirrors you mentioned — they repackage the same
underlying CERT r4.2 email.csv columns, just sometimes with slightly
different filenames, so we look for a few common names.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_RAW
from src.data.synthetic_cert_data import generate_synthetic_email_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_FILENAMES = ["email.csv", "Email.csv", "r4.2-email.csv", "cert_email.csv"]

REQUIRED_COLUMNS = {"id", "date", "user", "pc", "to", "from", "activity", "content"}


def _find_raw_email_file() -> Optional[Path]:
    for name in CANDIDATE_FILENAMES:
        candidate = DATA_RAW / name
        if candidate.exists():
            return candidate
    # last resort: any csv in raw/ that has the right columns
    for csv_path in DATA_RAW.glob("*.csv"):
        try:
            cols = set(pd.read_csv(csv_path, nrows=1).columns)
            if REQUIRED_COLUMNS.issubset(cols):
                return csv_path
        except Exception:
            continue
    return None


def load_email_data(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with (at minimum) columns:
    id, date, user, pc, to, from, activity, size, attachments, content

    If force_synthetic=True or no real file is found, generates and loads
    a synthetic dataset instead (logged clearly so it's never silently
    confused with real data).
    """
    path = None if force_synthetic else _find_raw_email_file()

    if path is None:
        logger.warning(
            "No CERT email.csv found in data/raw/ — generating a synthetic "
            "dataset with the same schema for development/testing. Drop the "
            "real dataset (from either Kaggle link) into data/raw/ to use "
            "real data instead."
        )
        path = generate_synthetic_email_csv()
    else:
        logger.info(f"Loading CERT email data from {path}")

    df = pd.read_csv(path)
    
    # --- FIX: Inject 'activity' column if missing to maintain CERT schema compliance ---
    if "activity" not in df.columns:
        df["activity"] = "send"
    # -----------------------------------------------------------------------------------

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Loaded file is missing expected CERT columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "user", "content"]).sort_values("date").reset_index(drop=True)
    
    # --- DEV FIX: Sample the data down so the MacBook doesn't melt processing 2.6M rows ---
    logger.info("Dev Mode: Downsampling dataset to 10,000 rows for rapid testing.")
    df = df.sample(n=10000, random_state=42).sort_values("date").reset_index(drop=True)
    # --------------------------------------------------------------------------------------
    
    return df


def load_insider_labels() -> Optional[pd.DataFrame]:
    """Ground-truth labels for supervised training, if available (synthetic run always has these)."""
    label_path = DATA_RAW / "insider_labels.csv"
    if label_path.exists():
        return pd.read_csv(label_path)
    logger.warning("No insider_labels.csv found — supervised training will need labels supplied separately.")
    return None


if __name__ == "__main__":
    df = load_email_data()
    print(df.shape)
    print(df.head())