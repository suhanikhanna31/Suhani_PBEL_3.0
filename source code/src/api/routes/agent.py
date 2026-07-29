"""
POST /api/agent/investigate/{pseudonym}

Runs the agentic investigation pipeline (src/agents/investigation_agent.py)
against one pseudonym and returns a structured, auditable report: the
evidence it gathered, whether watsonx.ai was used to reason over that
evidence, and a recommended next step.

This is read-only and additive to the rest of the API — it never mutates
risk data or consent state, and (like every other route here) always
requires an analyst/admin role token. The report it returns explicitly
carries `requires_human_review: true`; nothing in this route or the agent
it calls acts on the recommendation automatically.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from src.governance.access_control import get_current_role, check_permission
from src.agents.investigation_agent import investigate_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/investigate/{pseudonym}")
def run_investigation(pseudonym: str, role: str = Depends(get_current_role)):
    check_permission(role, "run_investigation")
    report = investigate_user(pseudonym, actor=role)
    if report.get("error"):
        raise HTTPException(status_code=404, detail=report["error"])
    return report
