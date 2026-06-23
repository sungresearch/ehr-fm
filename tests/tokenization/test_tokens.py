"""Tests for token string builders."""

from ehr_fm.tokenization.tokens import quantile_token, stage_token, text_token


class TestQuantileTokens:
    """Test quantile token builders."""

    def test_quantile_token_format(self):
        """Q:k format correct for various k."""
        assert quantile_token(1) == "Q:1"
        assert quantile_token(5) == "Q:5"
        assert quantile_token(10) == "Q:10"


class TestStageTokens:
    """Test stage token builders."""

    def test_stage_token_format(self):
        """STAGE:value format correct."""
        assert stage_token("order") == "STAGE:order"
        assert stage_token("taken") == "STAGE:taken"
        assert stage_token("admin") == "STAGE:admin"

    def test_stage_token_lowercased(self):
        """Stage value lowercased in token."""
        assert stage_token("ORDER") == "STAGE:order"
        assert stage_token("Taken") == "STAGE:taken"
        assert stage_token("ADMIN") == "STAGE:admin"


class TestTextTokens:
    """Test text token builders."""

    def test_text_token_format(self):
        """TXT:value format correct."""
        assert text_token("oral") == "TXT:oral"
        assert text_token("iv") == "TXT:iv"
        assert text_token("subcutaneous") == "TXT:subcutaneous"
