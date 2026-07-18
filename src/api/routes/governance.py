"""
GET  /api/governance/audit-log/verify -> tamper-evidence check on the hash-chained audit log
POST /api/governance/consent/revoke   -> revoke a user's consent (admin only)
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.governance.access_control import get_current_role, check_permission
from src.governance.audit_log import verify_chain, log_event
from src.governance.consent import revoke_consent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


class ConsentRevokeRequest(BaseModel):
    user_raw_identifier: str  # intentionally the RAW username, not pseudonym — see note below


@router.get("/audit-log/verify")
def verify_audit_log(role: str = Depends(get_current_role)):
    check_permission(role, "view_audit_log")
    is_valid = verify_chain()
    return {"chain_valid": is_valid}


@router.post("/consent/revoke")
def revoke_user_consent(req: ConsentRevokeRequest, role: str = Depends(get_current_role)):
    check_permission(role, "revoke_consent")
    # NOTE: consent revocation necessarily happens on the raw username (the
    # consent ledger predates pseudonymization — that's the whole point: a
    # user must be able to revoke consent by their real identity even though
    # everything downstream of that point only ever sees their pseudonym).
    revoke_consent(req.user_raw_identifier)
    log_event("consent_revoked", user_pseudonym=None,
              details={"note": "revoked by raw identifier, not logged here for privacy"}, actor=role)
    return {"status": "revoked", "note": "Re-run the pipeline for this to take effect on new data."}
