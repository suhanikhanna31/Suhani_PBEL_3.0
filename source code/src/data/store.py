"""
Storage helpers for pipeline outputs (scored_messages, user_risk).

Mirrors the same fallback pattern as src/data/ingest.py: when DATABASE_URL
is set, pipeline outputs are written to / read from live Neon Postgres
tables, so the deployed API (e.g. on Vercel, whose filesystem is read-only
and stateless between requests/deploys) can actually serve them. When
DATABASE_URL is unset, falls back to local CSVs under data/processed/, so
`python -m src.pipeline` + local `uvicorn` still works exactly as before
for anyone not using Neon.

This is why the "Pipeline has not been run yet" error shows up on Vercel
even after running the pipeline locally: local data/processed/*.csv files
never make it to the deployed function, since each serverless invocation
gets its own throwaway filesystem. Writing to Neon instead fixes that,
because Neon is a real persistent database reachable from anywhere.
"""
import logging
from typing import Optional

import pandas as pd

from src.config import DATA_PROCESSED, DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        return create_engine(DATABASE_URL)
    except ImportError:
        logger.warning(
            "DATABASE_URL is set but sqlalchemy/psycopg2-binary aren't "
            "installed (pip install -r requirements.txt) — falling back to "
            "local CSV storage under data/processed/."
        )
        return None


def write_df(df: pd.DataFrame, table_name: str) -> None:
    """
    Persists df as `table_name`. If DATABASE_URL is set, writes (replaces)
    a Neon Postgres table of the same name — this is what the deployed API
    reads from. Always ALSO writes a local CSV under data/processed/ so
    local development without Neon keeps working unchanged.
    """
    csv_path = DATA_PROCESSED / f"{table_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {len(df)} rows to local {csv_path}")

    engine = _get_engine()
    if engine is None:
        return
    try:
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(f"Wrote {len(df)} rows to Neon table '{table_name}'.")
    except Exception as e:
        logger.warning(
            f"Failed to write '{table_name}' to Neon ({e}) — local CSV was "
            "still saved, so local dev is unaffected."
        )


def read_df(table_name: str) -> Optional[pd.DataFrame]:
    """
    Reads `table_name`. Tries Neon first if DATABASE_URL is set; falls back
    to the local CSV under data/processed/ if Neon is unset/unreachable/the
    table doesn't exist there yet. Returns None if neither source has data
    (i.e. the pipeline genuinely hasn't been run anywhere yet).
    """
    engine = _get_engine()
    if engine is not None:
        try:
            df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
            return df
        except Exception as e:
            logger.warning(
                f"Could not read '{table_name}' from Neon ({e}) — falling "
                "back to local CSV."
            )

    csv_path = DATA_PROCESSED / f"{table_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None
