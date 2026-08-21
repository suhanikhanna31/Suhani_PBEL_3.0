"""
Tests for the agentic investigation module (src/agents/investigation_agent.py).
Exercises the rule-based fallback path only — no live watsonx.ai calls are
made in CI, matching the "degrades gracefully without credentials" design
used throughout this project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.agents import investigation_agent as agent


def _write_processed_csvs(tmp_path, avg_drift=1.0, flagged_rate=0.02):
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "user": "emp_test01",
        "avg_drift_score": avg_drift,
        "max_drift_score": avg_drift + 1,
        "n_messages": 50,
        "n_flagged_messages": 1,
        "flagged_message_rate": flagged_rate,
    }]).to_csv(tmp_path / "processed" / "user_risk.csv", index=False)

    pd.DataFrame([{
        "user": "emp_test01", "date": "2026-01-01", "drift_score": avg_drift,
        "n_flagged": 0, "flagged_features": "", "sentiment_polarity": 0.1,
        "urgency_score": 0.0,
    }]).to_csv(tmp_path / "processed" / "scored_messages.csv", index=False)
    return tmp_path / "processed"


class TestInvestigationAgentFallback:
    def test_unknown_user_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "DATA_PROCESSED", _write_processed_csvs(tmp_path))
        monkeypatch.setattr(agent, "_get_model", lambda: None)
        monkeypatch.setattr(agent, "explain_drift", lambda *a, **k: None)
        monkeypatch.setattr(agent, "log_event", lambda *a, **k: None)

        report = agent.investigate_user("nonexistent_user")
        assert "error" in report
        assert report["requires_human_review"] is True

    def test_low_drift_recommends_no_action(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "DATA_PROCESSED", _write_processed_csvs(tmp_path, avg_drift=0.5, flagged_rate=0.0))
        monkeypatch.setattr(agent, "_get_model", lambda: None)
        monkeypatch.setattr(agent, "explain_drift", lambda *a, **k: None)
        monkeypatch.setattr(agent, "get_recent_entries", lambda n=100: [])
        monkeypatch.setattr(agent, "log_event", lambda *a, **k: None)

        report = agent.investigate_user("emp_test01")
        assert report["recommendation"] == "no_action_needed"
        assert report["watsonx_reasoning_used"] is False
        assert report["requires_human_review"] is True

    def test_high_drift_recommends_escalation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "DATA_PROCESSED", _write_processed_csvs(tmp_path, avg_drift=5.0, flagged_rate=0.3))
        monkeypatch.setattr(agent, "_get_model", lambda: None)
        monkeypatch.setattr(agent, "explain_drift", lambda *a, **k: None)
        monkeypatch.setattr(agent, "get_recent_entries", lambda n=100: [])
        monkeypatch.setattr(agent, "log_event", lambda *a, **k: None)
        monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])

        report = agent.investigate_user("emp_test01")
        assert report["recommendation"] == "escalate_for_manual_review"
        assert report["requires_human_review"] is True

    def test_report_never_allows_autonomous_action(self, tmp_path, monkeypatch):
        """The requires_human_review flag must always be True — it is not
        something the evidence or a model response can flip."""
        monkeypatch.setattr(agent, "DATA_PROCESSED", _write_processed_csvs(tmp_path, avg_drift=9.0, flagged_rate=0.9))
        monkeypatch.setattr(agent, "_get_model", lambda: None)
        monkeypatch.setattr(agent, "explain_drift", lambda *a, **k: None)
        monkeypatch.setattr(agent, "get_recent_entries", lambda n=100: [])
        monkeypatch.setattr(agent, "log_event", lambda *a, **k: None)
        monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])

        report = agent.investigate_user("emp_test01")
        assert report["requires_human_review"] is True
