"""
PART 1 — Validation.

Lightweight schema + sanity checks run after ingest and after anonymize,
so a malformed upstream file fails loudly here instead of silently
producing garbage features three modules downstream.
"""
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["id", "date", "user", "content"]


class ValidationError(Exception):
    pass


def validate_email_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValidationError("DataFrame is empty after loading/filtering.")

    if df["content"].isna().any():
        n = df["content"].isna().sum()
        logger.warning(f"{n} rows have null content — these will be dropped by callers, not auto-fixed here.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValidationError("`date` column must be parsed to datetime before validation.")

    n_users = df["user"].nunique()
    if n_users < 2:
        logger.warning(f"Only {n_users} distinct user(s) present — baseline/drift comparisons need more users to be meaningful.")

    logger.info(f"Validation passed: {len(df)} rows, {n_users} users, "
                f"date range {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    from src.data.ingest import load_email_data
    df = load_email_data()
    validate_email_df(df)
