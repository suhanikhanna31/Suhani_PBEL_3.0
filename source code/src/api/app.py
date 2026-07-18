"""
PART 2 — FastAPI entrypoint.

Run locally with:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Serves both the JSON API (mounted routers) and the static frontend
dashboard (frontend/) so the whole thing is one deployable unit — matches
the IBM Code Engine deployment target (single container).
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import ROOT_DIR
from src.api.routes import users, drift, assistant, governance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insider Threat NLP — Behavioral Drift Detection API",
    description=(
        "Detects linguistic/behavioral drift in internal communications as a "
        "privacy-respecting insider-risk signal. All user-facing data is "
        "pseudonymous by design — see docs/ETHICS_AND_PRIVACY.md."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production to the actual frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(drift.router, prefix="/api/drift", tags=["drift"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["watsonx-assistant"])
app.include_router(governance.router, prefix="/api/governance", tags=["governance"])


@app.get("/health")
def health():
    return {"status": "ok"}


# Mounted LAST and at "/": StaticFiles(html=True) greedily serves "/" and any
# unmatched path as index.html, so it must be registered after every API
# route (and after /health) to avoid shadowing them.
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
