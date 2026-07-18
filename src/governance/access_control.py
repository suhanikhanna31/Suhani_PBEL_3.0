"""
PART 2 — Access control.

Simple role-based gate used by the FastAPI routes: only 'analyst' and
'admin' roles can view drift/risk data (still pseudonymous — this doesn't
grant re-identification), and only 'admin' can trigger consent revocation
or view audit-log internals. In production, swap `_ROLE_HEADER` checks for
real auth (IBM Cloud App ID / IAM-issued JWT validation) — the dependency
signature below is written so that swap doesn't touch route code.
"""
import logging
from fastapi import Header, HTTPException, status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEV-ONLY role map keyed by a bearer-ish header. Replace with real IAM token
# validation (e.g. IBM Cloud App ID) before any production use.
_DEV_ROLE_TOKENS = {
    "dev-analyst-token": "analyst",
    "dev-admin-token": "admin",
}

ROLE_PERMISSIONS = {
    "analyst": {"view_drift", "view_users", "view_dashboard"},
    "admin": {"view_drift", "view_users", "view_dashboard", "revoke_consent", "view_audit_log"},
}


def get_current_role(x_api_role_token: str = Header(default=None)) -> str:
    role = _DEV_ROLE_TOKENS.get(x_api_role_token)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Role-Token header. "
                   "(Dev tokens: 'dev-analyst-token', 'dev-admin-token' — replace with real IAM in production.)",
        )
    return role


def require_permission(permission: str):
    def _dependency(role: str = None):
        pass
    return _dependency


def check_permission(role: str, permission: str) -> None:
    if permission not in ROLE_PERMISSIONS.get(role, set()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{role}' lacks permission '{permission}'.",
        )
