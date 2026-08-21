"""
PART 1 — Ingestion.

Loads CERT-schema email data from data/raw/. If no real CERT csv is
present (e.g. you haven't downloaded it from Kaggle yet), falls back to
generating a synthetic dataset with the same schema so the pipeline is
runnable end-to-end during development.

Supports both Kaggle mirrors you mentioned — they repackage the same
underlying CERT r4.2 email.csv columns, just sometimes with slightly
different filenames, so we look for a few common names.

SCALE NOTE — why this file streams instead of pd.read_csv(path):
The real CERT r4.2 email.csv is on the order of 2.6M rows with a
full-text `content` column, which is several GB in memory once pandas
parses it. Reading it in one call blocks on parsing/allocating the
*entire* file before a single row is usable — which is exactly what
freezes a laptop. Since this project only ever needs a bounded,
representative sample (MAX_INGEST_ROWS, default 10,000 — plenty for
per-user baselines and drift scoring), we stream the file in chunks via
`chunksize=` and keep a running reservoir sample, so peak memory is
bounded by one chunk plus the sample, never the whole file.
"""
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_RAW, DATABASE_URL, MAX_INGEST_ROWS, INGEST_CHUNK_SIZE
from src.data.synthetic_cert_data import generate_synthetic_email_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_FILENAMES = ["email.csv", "Email.csv", "r4.2-email.csv", "cert_email.csv"]

REQUIRED_COLUMNS = {"id", "date", "user", "pc", "to", "from", "activity", "content"}

# Neon table columns use *_addr / *_code suffixes (user, to, from are
# reserved words in Postgres), so this maps them back to the CERT schema
# names the rest of the pipeline expects.
_NEON_COLUMN_ALIASES = {
    "user_code": "user",
    "to_addr": "to",
    "from_addr": "from",
}


def _load_from_neon(max_rows: int) -> Optional[pd.DataFrame]:
    """
    If DATABASE_URL is set (see .env / config.py), pulls a random sample of
    up to max_rows from the live Neon `emails` table instead of reading a
    local CSV. Returns None (never raises) if DATABASE_URL is unset or the
    connection/query fails, so callers can transparently fall back to the
    local-file / synthetic path below.
    """
    if not DATABASE_URL:
        return None

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        logger.warning(
            "DATABASE_URL is set but sqlalchemy/psycopg2-binary aren't installed "
            "(pip install -r requirements.txt) — falling back to local file/synthetic data."
        )
        return None

    try:
        engine = create_engine(DATABASE_URL)
        query = text(
            """
            SELECT id, date, user_code, pc, to_addr, cc, bcc, from_addr,
                   size, attachments, content
            FROM emails
            ORDER BY random()
            LIMIT :cap
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cap": max_rows})
        df = df.rename(columns=_NEON_COLUMN_ALIASES)
        logger.info(f"Loaded {len(df)} rows from Neon Postgres (DATABASE_URL set).")
        return df
    except Exception as e:
        logger.warning(
            f"DATABASE_URL is set but loading from Neon failed ({e}) — "
            "falling back to local file/synthetic data."
        )
        return None


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


def _reservoir_sample_csv(path: Path, max_rows: int, chunk_size: int,
                           random_state: int = 42) -> pd.DataFrame:
    """
    Streams `path` in chunks and maintains a fixed-size reservoir sample of
    at most `max_rows` rows, using Algorithm R (reservoir sampling) applied
    chunk-by-chunk. This gives a uniform random sample across the *whole*
    file — not just the first `max_rows` rows — while never holding more
    than one chunk + the reservoir in memory at once.

    For files with <= max_rows rows, this is equivalent to just reading
    the whole file (nothing is dropped).
    """
    rng = random.Random(random_state)
    reservoir: list = []
    n_seen = 0

    columns = None
    reader = pd.read_csv(path, chunksize=chunk_size, dtype=str, keep_default_na=False)
    for chunk in reader:
        if columns is None:
            columns = list(chunk.columns)
        # itertuples is materially faster than iterrows for row-by-row
        # access, which matters here since a 2.6M-row real CERT file means
        # this loop runs 2.6M times regardless of chunk_size.
        for row in chunk.itertuples(index=False, name=None):
            n_seen += 1
            if len(reservoir) < max_rows:
                reservoir.append(row)
            else:
                # replace a random existing element with probability max_rows/n_seen
                j = rng.randint(0, n_seen - 1)
                if j < max_rows:
                    reservoir[j] = row

    if n_seen > max_rows:
        logger.info(
            f"Sampled {max_rows} rows out of {n_seen} total via streaming "
            f"reservoir sampling (chunk_size={chunk_size}) — the full file "
            f"was never loaded into memory at once."
        )
    else:
        logger.info(f"File has {n_seen} rows (<= {max_rows} cap) — loaded in full via streaming.")

    df = pd.DataFrame(reservoir, columns=columns)
    return df


def load_email_data(force_synthetic: bool = False,
                     max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Returns a DataFrame with (at minimum) columns:
    id, date, user, pc, to, from, activity, size, attachments, content

    If force_synthetic=True or no real file is found, generates and loads
    a synthetic dataset instead (logged clearly so it's never silently
    confused with real data).

    max_rows caps how many rows are read from a *real* dataset, via
    streaming reservoir sampling (see _reservoir_sample_csv) so arbitrarily
    large files never need to fit in memory at once. Defaults to
    config.MAX_INGEST_ROWS (10,000) — override per-call, or via the
    MAX_INGEST_ROWS env var, if you have the RAM for more.
    """
    cap = MAX_INGEST_ROWS if max_rows is None else max_rows

    df = None if force_synthetic else _load_from_neon(max_rows=cap)

    if df is None:
        path = None if force_synthetic else _find_raw_email_file()

        if path is None:
            logger.warning(
                "No DATABASE_URL set and no CERT email.csv found in data/raw/ — "
                "generating a synthetic dataset with the same schema for "
                "development/testing. Set DATABASE_URL to your Neon connection "
                "string, or drop the real dataset into data/raw/, to use real "
                "data instead."
            )
            path = generate_synthetic_email_csv()
            df = pd.read_csv(path)
        else:
            logger.info(f"Loading CERT email data from {path} (streaming, cap={cap} rows)")
            df = _reservoir_sample_csv(path, max_rows=cap, chunk_size=INGEST_CHUNK_SIZE)

    # Some real-world CERT mirrors drop the `activity` column (it's not
    # used by any feature extractor here, only kept for schema parity), so
    # backfill a constant default rather than failing ingestion over an
    # unused field.
    if "activity" not in df.columns:
        logger.info("`activity` column absent from source file — defaulting to 'send' (schema parity only, unused downstream).")
        df["activity"] = "send"

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Loaded file is missing expected CERT columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "user", "content"]).sort_values("date").reset_index(drop=True)

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
