"""Tests for numeric feature construction functions."""

import math

import pytest

from ehr_fm.pretokenize.embedding_numeric import (
    LOG1P_SCALE_CONSTANT,
    MIN_REF_RANGE_WIDTH,
    _compute_numeric_features,
    _compute_numeric_features_ref_range_priority,
)


class TestRefRangePriority:
    """Tests for _compute_numeric_features_ref_range_priority.

    Output layout: [x_primary, is_refrange, is_log1p, value_present]
    """

    # -- absent / non-finite --------------------------------------------------

    def test_null_value(self):
        event = {"numeric_value": None}
        assert _compute_numeric_features_ref_range_priority(event) == [0.0, 0.0, 0.0, 0.0]

    def test_missing_numeric_value_key(self):
        event = {"code": "LAB/123"}
        assert _compute_numeric_features_ref_range_priority(event) == [0.0, 0.0, 0.0, 0.0]

    def test_nan_value(self):
        event = {"numeric_value": float("nan")}
        assert _compute_numeric_features_ref_range_priority(event) == [0.0, 0.0, 0.0, 0.0]

    def test_inf_value(self):
        event = {"numeric_value": float("inf")}
        assert _compute_numeric_features_ref_range_priority(event) == [0.0, 0.0, 0.0, 0.0]

    def test_negative_inf_value(self):
        event = {"numeric_value": float("-inf")}
        assert _compute_numeric_features_ref_range_priority(event) == [0.0, 0.0, 0.0, 0.0]

    # -- ref_range path --------------------------------------------------------

    def test_ref_range_path_normal(self):
        event = {"numeric_value": 5.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == pytest.approx(0.5)  # midpoint
        assert result[1] == 1.0  # is_refrange
        assert result[2] == 0.0  # is_log1p
        assert result[3] == 1.0  # value_present

    def test_ref_range_path_low_boundary(self):
        event = {"numeric_value": 4.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == 1.0

    def test_ref_range_path_high_boundary(self):
        event = {"numeric_value": 6.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == 1.0

    def test_ref_range_path_below_range_clipped(self):
        """Values far below ref_low should be clipped to -2.0."""
        event = {"numeric_value": -10.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == -2.0
        assert result[1] == 1.0

    def test_ref_range_path_above_range_clipped(self):
        """Values far above ref_high should be clipped to 3.0."""
        event = {"numeric_value": 100.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == 3.0
        assert result[1] == 1.0

    # -- raw log1p path --------------------------------------------------------

    def test_raw_log1p_path_no_ref_range(self):
        event = {"numeric_value": 100.0}
        result = _compute_numeric_features_ref_range_priority(event)
        expected = math.log1p(100.0) / LOG1P_SCALE_CONSTANT
        assert result[0] == pytest.approx(expected)
        assert result[1] == 0.0  # is_refrange
        assert result[2] == 1.0  # is_log1p
        assert result[3] == 1.0  # value_present

    def test_raw_log1p_path_ref_low_only(self):
        """Only ref_low present -> falls back to raw log1p."""
        event = {"numeric_value": 5.0, "ref_low": 3.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_raw_log1p_path_ref_high_only(self):
        """Only ref_high present -> falls back to raw log1p."""
        event = {"numeric_value": 5.0, "ref_high": 10.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_raw_log1p_path_equal_ref_bounds(self):
        """ref_low == ref_high -> falls back to raw log1p (avoids division by zero)."""
        event = {"numeric_value": 5.0, "ref_low": 5.0, "ref_high": 5.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_raw_log1p_path_inverted_ref_bounds(self):
        """ref_low > ref_high -> falls back to raw log1p."""
        event = {"numeric_value": 5.0, "ref_low": 10.0, "ref_high": 3.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_raw_log1p_path_clipped(self):
        """Extreme values should be clipped to 1.5."""
        event = {"numeric_value": 1e10}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == 1.5
        assert result[2] == 1.0

    def test_raw_log1p_path_negative_value(self):
        """Negative values use abs() before log1p."""
        event = {"numeric_value": -100.0}
        result = _compute_numeric_features_ref_range_priority(event)
        expected = math.log1p(100.0) / LOG1P_SCALE_CONSTANT
        assert result[0] == pytest.approx(expected)
        assert result[2] == 1.0

    def test_raw_log1p_path_zero_value(self):
        event = {"numeric_value": 0.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[0] == pytest.approx(0.0)  # log1p(0) = 0
        assert result[2] == 1.0
        assert result[3] == 1.0

    # -- degenerate ref ranges -------------------------------------------------

    def test_degenerate_ref_range_narrow(self):
        """Ref range narrower than MIN_REF_RANGE_WIDTH falls back to log1p."""
        width = MIN_REF_RANGE_WIDTH / 2
        event = {"numeric_value": 100.0, "ref_low": 100.0, "ref_high": 100.0 + width}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0  # NOT ref_range path
        assert result[2] == 1.0  # log1p path

    def test_ref_range_at_min_width(self):
        """Ref range exactly at MIN_REF_RANGE_WIDTH is borderline (exclusive threshold)."""
        event = {"numeric_value": 0.005, "ref_low": 0.0, "ref_high": MIN_REF_RANGE_WIDTH}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0  # not > MIN_REF_RANGE_WIDTH, so log1p

    def test_ref_range_just_above_min_width(self):
        """Ref range just above MIN_REF_RANGE_WIDTH uses ref_range path."""
        width = MIN_REF_RANGE_WIDTH * 2
        event = {"numeric_value": 100.0, "ref_low": 100.0, "ref_high": 100.0 + width}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 1.0  # ref_range path

    def test_nan_ref_low(self):
        """NaN ref_low falls back to log1p."""
        event = {"numeric_value": 5.0, "ref_low": float("nan"), "ref_high": 10.0}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_nan_ref_high(self):
        """NaN ref_high falls back to log1p."""
        event = {"numeric_value": 5.0, "ref_low": 3.0, "ref_high": float("nan")}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_inf_ref_bounds(self):
        """Inf ref bounds fall back to log1p."""
        event = {"numeric_value": 5.0, "ref_low": 0.0, "ref_high": float("inf")}
        result = _compute_numeric_features_ref_range_priority(event)
        assert result[1] == 0.0
        assert result[2] == 1.0

    # -- structural invariants -------------------------------------------------

    def test_output_always_length_4(self):
        """All branches produce exactly 4-element vectors."""
        cases = [
            {"numeric_value": None},
            {"numeric_value": float("nan")},
            {"numeric_value": 5.0},
            {"numeric_value": 5.0, "ref_low": 3.0, "ref_high": 7.0},
        ]
        for event in cases:
            result = _compute_numeric_features_ref_range_priority(event)
            assert len(result) == 4, f"Expected 4 elements for {event}, got {len(result)}"

    def test_exactly_one_path_flag_when_present(self):
        """When value_present=1, exactly one of is_refrange/is_log1p is 1."""
        cases = [
            {"numeric_value": 5.0},
            {"numeric_value": 5.0, "ref_low": 3.0, "ref_high": 7.0},
            {"numeric_value": -10.0, "ref_low": 3.0, "ref_high": 7.0},
            {"numeric_value": 0.0},
        ]
        for event in cases:
            result = _compute_numeric_features_ref_range_priority(event)
            assert result[3] == 1.0  # value_present
            assert result[1] + result[2] == 1.0, f"Flags sum != 1 for {event}: {result}"

    def test_no_flags_when_absent(self):
        """When value is absent, all elements are 0."""
        for event in [{"numeric_value": None}, {"numeric_value": float("nan")}]:
            result = _compute_numeric_features_ref_range_priority(event)
            assert result == [0.0, 0.0, 0.0, 0.0]


class TestLegacyZscoreUnchanged:
    """Sanity check that _compute_numeric_features still produces 5-dim vectors."""

    def test_null_value(self):
        event = {"numeric_value": None}
        result = _compute_numeric_features(event, "LAB/123", {}, {})
        assert result == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_with_ref_range(self):
        event = {"numeric_value": 5.0, "ref_low": 4.0, "ref_high": 6.0}
        result = _compute_numeric_features(event, "LAB/123", {}, {})
        assert len(result) == 5
        assert result[4] == 1.0  # value_present
        assert result[3] == 1.0  # ref_range_available

    def test_without_ref_range(self):
        event = {"numeric_value": 5.0}
        result = _compute_numeric_features(event, "LAB/123", {}, {})
        assert len(result) == 5
        assert result[4] == 1.0  # value_present
        assert result[3] == 0.0  # ref_range_available
