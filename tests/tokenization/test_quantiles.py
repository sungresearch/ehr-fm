"""Tests for quantile bucket assignment."""

import pytest

from ehr_fm.tokenization.quantiles import assign_quantile_bucket


class TestAssignQuantileBucket:
    """Test quantile bucket assignment."""

    @pytest.fixture
    def sample_breaks(self):
        """Sample breaks for LAB/glucose: 5 buckets with 4 break points."""
        return {"LAB/glucose": [70.0, 85.0, 100.0, 120.0]}

    def test_first_bucket(self, sample_breaks):
        """Value below first break → Q:1."""
        result = assign_quantile_bucket("LAB/glucose", 60.0, sample_breaks)
        assert result == "Q:1"

    def test_last_bucket(self, sample_breaks):
        """Value above last break → Q:5."""
        result = assign_quantile_bucket("LAB/glucose", 150.0, sample_breaks)
        assert result == "Q:5"

    def test_middle_bucket(self, sample_breaks):
        """Value between breaks → correct bucket."""
        # Between 70 and 85 → Q:2
        result = assign_quantile_bucket("LAB/glucose", 75.0, sample_breaks)
        assert result == "Q:2"

        # Between 85 and 100 → Q:3
        result = assign_quantile_bucket("LAB/glucose", 90.0, sample_breaks)
        assert result == "Q:3"

        # Between 100 and 120 → Q:4
        result = assign_quantile_bucket("LAB/glucose", 110.0, sample_breaks)
        assert result == "Q:4"

    def test_exact_break_value(self, sample_breaks):
        """Value exactly on break → next bucket (>= comparison)."""
        # Exactly at 70 → Q:2 (>= first break)
        result = assign_quantile_bucket("LAB/glucose", 70.0, sample_breaks)
        assert result == "Q:2"

        # Exactly at 100 → Q:4 (>= third break)
        result = assign_quantile_bucket("LAB/glucose", 100.0, sample_breaks)
        assert result == "Q:4"

    def test_missing_code(self, sample_breaks):
        """Code not in breaks → Q:UNK."""
        result = assign_quantile_bucket("LAB/unknown", 100.0, sample_breaks)
        assert result == "Q:UNK"

    def test_empty_breaks_dict(self):
        """Empty breaks dict → Q:UNK."""
        result = assign_quantile_bucket("LAB/glucose", 100.0, {})
        assert result == "Q:UNK"

    def test_empty_breaks_for_code(self):
        """Code with empty breaks list → Q:1 (invariant values).

        When a code exists in the breaks dict but has empty breaks list,
        it means all training samples had identical values (invariant).
        All values should map to bucket 1 in this case.
        """
        breaks = {"LAB/glucose": []}
        result = assign_quantile_bucket("LAB/glucose", 100.0, breaks)
        assert result == "Q:1"

    def test_invariant_values_semantic(self):
        """Verify semantic difference between missing code and invariant values.

        - Code NOT in dict → Q:UNK (never seen during training)
        - Code in dict with [] → Q:1 (seen, but all values identical)
        """
        # Code seen during training but had invariant values
        breaks = {"invariant_code": [], "normal_code": [50.0, 100.0]}

        # Invariant code → Q:1 (all values map to bucket 1)
        assert assign_quantile_bucket("invariant_code", 0.0, breaks) == "Q:1"
        assert assign_quantile_bucket("invariant_code", 1000.0, breaks) == "Q:1"

        # Missing code → Q:UNK
        assert assign_quantile_bucket("never_seen_code", 50.0, breaks) == "Q:UNK"

        # Normal code with breaks → proper bucket assignment
        assert assign_quantile_bucket("normal_code", 25.0, breaks) == "Q:1"
        assert assign_quantile_bucket("normal_code", 75.0, breaks) == "Q:2"
        assert assign_quantile_bucket("normal_code", 150.0, breaks) == "Q:3"

    def test_single_break(self):
        """Single break value → Q:1 or Q:2."""
        breaks = {"LAB/test": [100.0]}

        # Below break → Q:1
        result = assign_quantile_bucket("LAB/test", 50.0, breaks)
        assert result == "Q:1"

        # At or above break → Q:2
        result = assign_quantile_bucket("LAB/test", 100.0, breaks)
        assert result == "Q:2"

    def test_negative_values(self):
        """Negative numeric values handled."""
        breaks = {"LAB/test": [-50.0, 0.0, 50.0]}

        result = assign_quantile_bucket("LAB/test", -100.0, breaks)
        assert result == "Q:1"

        result = assign_quantile_bucket("LAB/test", -25.0, breaks)
        assert result == "Q:2"

        result = assign_quantile_bucket("LAB/test", 25.0, breaks)
        assert result == "Q:3"

        result = assign_quantile_bucket("LAB/test", 100.0, breaks)
        assert result == "Q:4"

    def test_large_values(self):
        """Very large numeric values → last bucket."""
        breaks = {"LAB/test": [100.0]}
        result = assign_quantile_bucket("LAB/test", 1e10, breaks)
        assert result == "Q:2"

    def test_zero_value(self):
        """Zero numeric value handled."""
        breaks = {"LAB/test": [-10.0, 10.0]}

        result = assign_quantile_bucket("LAB/test", 0.0, breaks)
        assert result == "Q:2"  # >= -10, but < 10
