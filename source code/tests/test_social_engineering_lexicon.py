"""
Unit tests for src/dsa/social_engineering_lexicon.py.

Kept in a separate file from tests/test_core.py rather than appended to it,
so the existing test file has zero diff and this feature's tests are
independently reviewable/runnable, matching the project's existing pattern
of one test module per feature area.
"""
import pytest

from src.dsa.social_engineering_lexicon import SocialEngineeringLexicon


class TestSocialEngineeringLexicon:
    def test_category_scores_returns_all_categories(self):
        lex = SocialEngineeringLexicon()
        scores = lex.category_scores("just a normal message about the weekly report")
        assert set(scores.keys()) == {
            "authority_spoofing",
            "isolation_secrecy",
            "artificial_scarcity",
            "trust_exploitation",
            "curiosity_clickbait",
        }
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_authority_spoofing_match(self):
        lex = SocialEngineeringLexicon()
        scores = lex.category_scores("This is a direct order from the CEO.")
        assert scores["authority_spoofing"] > 0.0

    def test_isolation_secrecy_match(self):
        lex = SocialEngineeringLexicon()
        scores = lex.category_scores("Please keep this strictly between us.")
        assert scores["isolation_secrecy"] > 0.0

    def test_no_match_scores_zero(self):
        lex = SocialEngineeringLexicon()
        scores = lex.category_scores("The quarterly report is attached for review.")
        assert all(v == 0.0 for v in scores.values())

    def test_social_engineering_score_bounded(self):
        lex = SocialEngineeringLexicon()
        score = lex.social_engineering_score(
            "Direct order from the CEO — keep this strictly between us, "
            "only available today, as a valued colleague, click to see."
        )
        assert 0.0 <= score <= 1.0

    def test_social_engineering_score_zero_for_empty_text(self):
        lex = SocialEngineeringLexicon()
        assert lex.social_engineering_score("") == 0.0

    def test_scan_returns_per_category_matches(self):
        lex = SocialEngineeringLexicon()
        matches = lex.scan("This is a direct order from the CEO.")
        assert any(matches["authority_spoofing"])
        assert not matches["isolation_secrecy"]

    def test_custom_lexicon_override(self):
        custom = {"test_category": ["special phrase"]}
        lex = SocialEngineeringLexicon(lexicon=custom)
        scores = lex.category_scores("this contains a special phrase in it")
        assert scores == {"test_category": pytest.approx(scores["test_category"])}
        assert scores["test_category"] > 0.0
