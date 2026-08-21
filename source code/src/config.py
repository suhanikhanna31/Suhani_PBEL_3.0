"""
Central configuration for the insider-threat-nlp project.
Reads from environment variables (see .env.example). Keeping this in one
place means every module (ingest, features, api, governance) agrees on
paths, salts, and toggles instead of hardcoding them
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_INTERIM = ROOT_DIR / "data" / "interim"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Database (Neon Postgres) ----
# When set, ingest.py reads email data from this live Postgres database
# (the `emails` table) instead of a local data/raw/email.csv file. Leave
# unset to keep using local CSV / synthetic data as before — this is a
# fallback chain, not a hard requirement.
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ---- watsonx.ai ----
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
WATSONX_ENABLED = bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)

# ---- watsonx Assistant ----
ASSISTANT_WEBHOOK_SECRET = os.getenv("ASSISTANT_WEBHOOK_SECRET", "change-me")

# ---- Privacy / governance ----
ANONYMIZATION_SALT = os.getenv("ANONYMIZATION_SALT", "dev-salt-change-me")
CONSENT_REQUIRED = os.getenv("CONSENT_REQUIRED", "true").lower() == "true"
AUDIT_LOG_PATH = ROOT_DIR / os.getenv("AUDIT_LOG_PATH", "data/processed/audit_log.jsonl")

# ---- Baseline / drift tuning ----
BASELINE_WINDOW_SIZE = int(os.getenv("BASELINE_WINDOW_SIZE", "30"))   # rolling window, in messages
DRIFT_Z_THRESHOLD = float(os.getenv("DRIFT_Z_THRESHOLD", "2.5"))      # z-score flagged as drift
TOP_K_RISKIEST = int(os.getenv("TOP_K_RISKIEST", "10"))

# ---- Ingestion scale controls ----
# The real CERT r4.2 email.csv is multiple GB / ~2.6M rows. Loading that
# with a single pd.read_csv() call reads the *entire* file (including the
# full-text `content` column) into memory before anything else can happen,
# which is what freezes a laptop. To keep this project runnable on
# consumer hardware, ingestion streams the file in chunks and keeps only a
# bounded, reproducible sample — it never materializes the full file in
# memory. Raise MAX_INGEST_ROWS (and give the process more RAM) if you
# want to work with more data than this.
MAX_INGEST_ROWS = int(os.getenv("MAX_INGEST_ROWS", "10000"))
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "50000"))

# ---- API ----
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---- Urgency / social-engineering lexicon (used by trie_phrase_matcher) ----
URGENCY_PHRASES = [
    "urgent", "asap", "right away", "immediately", "act now",
    "do not tell", "don't tell anyone", "keep this between us",
    "wire transfer", "gift card", "verify your password",
    "click here now", "your account will be suspended",
    "final notice", "before end of day", "confidential request",
    "bypass the process", "skip approval", "reset your credentials",
]

# ---- Categorized social-engineering lexicon (used by
# src/dsa/social_engineering_lexicon.py) ----
# Grouped by manipulation technique rather than left as one flat list, so
# a flagged message can be explained as "authority spoofing" or "artificial
# scarcity" rather than just "urgency" — see
# docs/ATTENTION_WARFARE_LEXICON.md for where each category comes from.
# Deliberately non-overlapping with URGENCY_PHRASES above; that lexicon and
# this one run as independent features (urgency_score vs
# social_engineering_score), not a replacement of one by the other.
SOCIAL_ENGINEERING_LEXICON = {
    "authority_spoofing": [
        "direct order from", "on behalf of the director", "ceo needs this",
        "compliance requires you to", "legal has approved this request",
        "executive request", "per instructions from senior management",
    ],
    "isolation_secrecy": [
        "do not forward this email", "keep this strictly between us",
        "do not loop in your manager", "this is off the record",
        "handle this discreetly", "do not cc anyone else",
    ],
    "artificial_scarcity": [
        "only available today", "last chance to act",
        "limited time offer", "window closes in one hour",
        "one time exception", "offer expires shortly",
    ],
    "trust_exploitation": [
        "as a valued colleague", "i know i can trust you with this",
        "just between friends", "you're the only one i can ask",
        "i trust you to handle this quietly",
    ],
    "curiosity_clickbait": [
        "you won't believe what", "click to see", "see what happened next",
        "check this out before it's too late", "you need to see this now",
    ],
}
