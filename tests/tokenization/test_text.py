"""Tests for text normalization."""

from ehr_fm.tokenization.text import make_text_token, normalize_text


class TestNormalizeText:
    """Test text normalization."""

    def test_lowercase(self):
        """Text converted to lowercase."""
        assert normalize_text("ORAL") == "oral"
        assert normalize_text("Intravenous") == "intravenous"

    def test_strip_whitespace(self):
        """Leading/trailing whitespace removed."""
        assert normalize_text("  oral  ") == "oral"
        assert normalize_text("\tiv\n") == "iv"

    def test_spaces_to_underscores(self):
        """Internal spaces → underscores."""
        assert normalize_text("oral route") == "oral_route"
        assert normalize_text("by mouth") == "by_mouth"

    def test_special_chars_removed(self):
        """Non-alphanumeric chars removed (except underscore)."""
        assert normalize_text("oral (by mouth)") == "oral_by_mouth"
        assert normalize_text("IV/SC") == "ivsc"
        assert normalize_text("test@value!") == "testvalue"

    def test_empty_string(self):
        """Empty string → None."""
        assert normalize_text("") is None

    def test_whitespace_only(self):
        """Whitespace-only string → None."""
        assert normalize_text("   ") is None
        assert normalize_text("\t\n") is None

    def test_special_chars_only(self):
        """Special-chars-only string → None."""
        assert normalize_text("@#$%") is None
        assert normalize_text("...") is None

    def test_multiple_spaces(self):
        """Multiple consecutive spaces → single underscore."""
        assert normalize_text("oral    route") == "oral_route"
        assert normalize_text("by   mouth") == "by_mouth"

    def test_already_normalized(self):
        """Already normalized text unchanged."""
        assert normalize_text("oral") == "oral"
        assert normalize_text("oral_route") == "oral_route"

    def test_preserves_numbers(self):
        """Numbers preserved in output."""
        assert normalize_text("dose 500mg") == "dose_500mg"
        assert normalize_text("100ml") == "100ml"


class TestMakeTextToken:
    """Test text token creation."""

    def test_valid_text(self):
        """Valid text → TXT:normalized."""
        assert make_text_token("oral") == "TXT:oral"
        assert make_text_token("ORAL") == "TXT:oral"
        assert make_text_token("by mouth") == "TXT:by_mouth"

    def test_empty_after_normalize(self):
        """Text that normalizes to empty → None."""
        assert make_text_token("") is None
        assert make_text_token("   ") is None
        assert make_text_token("@#$") is None

    def test_preserves_underscores(self):
        """Existing underscores preserved."""
        assert make_text_token("oral_route") == "TXT:oral_route"
