"""
Central configuration for the insider-threat-nlp project.
Reads from environment variables (see .env.example). Keeping this in one
place means every module (ingest, features, api, governance) agrees on
paths, salts, and toggles instead of hardcoding them.
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

# ---- watsonx.ai ----
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")
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
