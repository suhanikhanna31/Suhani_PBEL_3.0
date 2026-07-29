"""
IBM QRadar (SIEM) integration — CEF export.

This project's signal (linguistic drift) is currently a standalone
dashboard. In a real security operations setup, that signal is much more
useful sitting alongside the log/network-based alerts a SOC analyst
already triages in IBM QRadar (or any SIEM that speaks CEF), rather than
in a second dashboard nobody checks.

This module is an honestly-labeled INTEGRATION STUB: it converts flagged
users/messages into standard CEF (Common Event Format) strings, which is
the format QRadar (and most SIEMs) ingest over syslog. It does not open a
network connection to a real QRadar instance — this repo has no QRadar
credentials or test instance to integrate against. Wiring `send_to_qradar`
up to a real syslog/HEC endpoint is a small, well-defined next step once
those exist (see the TODO in that function).

CEF format reference: `CEF:Version|Vendor|Product|Version|SignatureID|
Name|Severity|Extension`
"""
import logging
from datetime import datetime, timezone
from typing import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CEF_VENDOR = "Signal-InsiderThreatNLP"
CEF_PRODUCT = "LinguisticDriftDetector"
CEF_VERSION = "1.0"


def user_risk_row_to_cef(row: dict) -> str:
    """
    Converts one row of the per-user risk table (pseudonymized user,
    avg_drift_score, flagged_message_rate, ...) into a single CEF event
    string suitable for a syslog line QRadar can parse.

    Severity is derived from avg_drift_score on a simple 0-10 scale
    clamped to CEF's expected 0-10 range — tune the scaling to your own
    drift-score distribution before using this in a real SIEM feed.
    """
    user = row.get("user", "unknown")
    avg_drift = float(row.get("avg_drift_score", 0.0))
    flagged_rate = float(row.get("flagged_message_rate", 0.0))
    n_messages = int(row.get("n_messages", 0))

    severity = max(0, min(10, round(avg_drift * 2)))

    extension = (
        f"suser={user} "
        f"cs1Label=avgDriftScore cs1={avg_drift} "
        f"cs2Label=flaggedMessageRate cs2={flagged_rate} "
        f"cnt={n_messages} "
        f"rt={datetime.now(timezone.utc).isoformat()}"
    )

    return (
        f"CEF:0|{CEF_VENDOR}|{CEF_PRODUCT}|{CEF_VERSION}|"
        f"LINGUISTIC_DRIFT|Insider-risk linguistic drift flagged|{severity}|{extension}"
    )


def export_user_risk_to_cef(user_risk_rows: Iterable[dict]) -> list:
    """Batch version of user_risk_row_to_cef — one CEF line per flagged user."""
    return [user_risk_row_to_cef(row) for row in user_risk_rows]


def send_to_qradar(cef_lines: list, host: str = None, port: int = 514) -> None:
    """
    STUB — not implemented against a live QRadar instance.

    In a real deployment this would open a syslog (UDP/TCP, or TLS syslog)
    connection to QRadar's log source and write each CEF line, e.g. via
    Python's `logging.handlers.SysLogHandler` pointed at QRadar's
    listener port. Left unimplemented here because doing it "for real"
    needs a live QRadar log source configured with real credentials/host,
    which this environment doesn't have.
    """
    raise NotImplementedError(
        "send_to_qradar is an integration stub — point it at a real QRadar "
        "syslog listener (host/port) and swap this for a SysLogHandler-based "
        "sender once one is available. export_user_risk_to_cef() already "
        "produces valid CEF lines you can pipe to `logger`/`nc`/a syslog "
        "forwarder manually in the meantime."
    )


if __name__ == "__main__":
    demo_row = {"user": "emp_a1b2c3d4e5f6", "avg_drift_score": 3.4,
                "flagged_message_rate": 0.22, "n_messages": 87}
    print(user_risk_row_to_cef(demo_row))
