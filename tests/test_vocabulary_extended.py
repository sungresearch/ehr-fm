"""Tests for vocabulary.py helpers: OnlineStatistics, ReservoirSampler, prepend_* functions."""

import math
import statistics

import pytest

from ehr_fm.defaults import DEFAULT_DEMOGRAPHIC_LABELS, DEFAULT_INTERVAL_LABELS
from ehr_fm.vocabulary import (
    OnlineStatistics,
    ReservoirSampler,
    prepend_demographic_tokens,
    prepend_interval_tokens,
)

# ===================================================================
# OnlineStatistics
# ===================================================================


class TestOnlineStatistics:
    def test_single_update(self):
        s = OnlineStatistics()
        s.update(5.0)
        assert s.mean == pytest.approx(5.0)
        assert s.count == pytest.approx(1.0)

    def test_multiple_updates(self):
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        s = OnlineStatistics()
        for v in values:
            s.update(v)
        assert s.mean == pytest.approx(statistics.mean(values))
        assert s.variance == pytest.approx(statistics.pvariance(values), rel=0.01)

    def test_zero_count_variance(self):
        s = OnlineStatistics()
        assert s.variance == 0.0

    def test_weighted_mean(self):
        s = OnlineStatistics()
        s.update(10.0, w=3.0)
        s.update(20.0, w=1.0)
        expected_mean = (10.0 * 3 + 20.0 * 1) / 4.0
        assert s.mean == pytest.approx(expected_mean)

    def test_std_property(self):
        s = OnlineStatistics()
        s.update(0.0)
        s.update(10.0)
        assert s.std == pytest.approx(math.sqrt(s.variance))

    def test_identical_values_zero_variance(self):
        s = OnlineStatistics()
        for _ in range(100):
            s.update(42.0)
        assert s.variance == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# ReservoirSampler
# ===================================================================


class TestReservoirSampler:
    def test_fill_phase(self):
        rs = ReservoirSampler(size=5, seed=0)
        for i in range(5):
            rs.update(float(i))
        assert rs.samples == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_overflow_keeps_size(self):
        rs = ReservoirSampler(size=3, seed=42)
        for i in range(100):
            rs.update(float(i))
        assert len(rs.samples) == 3
        assert rs.total_weight == pytest.approx(100.0)

    def test_seeded_reproducibility(self):
        def _run(seed: int) -> list[float]:
            rs = ReservoirSampler(size=5, seed=seed)
            for i in range(200):
                rs.update(float(i))
            return rs.samples[:]

        a = _run(123)
        b = _run(123)
        assert a == b

    def test_large_stream_coverage(self):
        n = 10_000
        rs = ReservoirSampler(size=100, seed=7)
        for i in range(n):
            rs.update(float(i))

        mean_sample = statistics.mean(rs.samples)
        expected_mid = n / 2.0
        assert abs(mean_sample - expected_mid) < n * 0.6

    def test_partial_fill(self):
        rs = ReservoirSampler(size=10, seed=0)
        rs.update(1.0)
        rs.update(2.0)
        assert rs.n == 2
        assert rs.samples[0] == 1.0
        assert rs.samples[1] == 2.0
        assert math.isnan(rs.samples[2])


# ===================================================================
# prepend_interval_tokens
# ===================================================================


class TestPrependIntervalTokens:
    def test_prepends_default_count(self):
        base = [{"type": "code", "label": "X", "weight": 1.0}]
        target = len(base) + len(DEFAULT_INTERVAL_LABELS)
        result = prepend_interval_tokens(base, target_vocab_size=target)
        interval_entries = [e for e in result if e.get("type") == "interval"]
        assert len(interval_entries) == len(DEFAULT_INTERVAL_LABELS)

    def test_deduplication(self):
        existing = [{"type": "interval", "label": DEFAULT_INTERVAL_LABELS[0], "weight": -1.0}]
        target = 1 + len(DEFAULT_INTERVAL_LABELS)
        result = prepend_interval_tokens(existing, target_vocab_size=target)
        labels = [e["label"] for e in result if e.get("type") == "interval"]
        assert len(labels) == len(set(labels))

    def test_default_truncates_to_original_size(self):
        base = [{"type": "code", "label": f"C{i}", "weight": 1.0} for i in range(20)]
        result = prepend_interval_tokens(base)
        assert len(result) == 20

    def test_explicit_truncation(self):
        base = [{"type": "code", "label": f"C{i}", "weight": 1.0} for i in range(100)]
        result = prepend_interval_tokens(base, target_vocab_size=20)
        assert len(result) == 20

    def test_no_truncation_when_combined_fits(self):
        base = [{"type": "code", "label": f"C{i}", "weight": 1.0} for i in range(5)]
        big_target = 500
        result = prepend_interval_tokens(base, target_vocab_size=big_target)
        assert len(result) == len(DEFAULT_INTERVAL_LABELS) + len(base)

    def test_intervals_are_first(self):
        base = [{"type": "code", "label": "X", "weight": 1.0}]
        target = 1 + len(DEFAULT_INTERVAL_LABELS)
        result = prepend_interval_tokens(base, target_vocab_size=target)
        assert result[0]["type"] == "interval"


# ===================================================================
# prepend_demographic_tokens
# ===================================================================


class TestPrependDemographicTokens:
    def test_prepends_demographic_labels(self):
        base = [{"type": "code", "label": "X", "weight": 1.0}]
        target = len(base) + len(DEFAULT_DEMOGRAPHIC_LABELS)
        result = prepend_demographic_tokens(base, target_vocab_size=target)
        demo_entries = [e for e in result if e.get("type") == "demographic"]
        assert len(demo_entries) == len(DEFAULT_DEMOGRAPHIC_LABELS)

    def test_deduplication(self):
        existing = [{"type": "demographic", "label": "DEMO_SEX_MALE", "weight": -1.0}]
        target = 1 + len(DEFAULT_DEMOGRAPHIC_LABELS)
        result = prepend_demographic_tokens(existing, target_vocab_size=target)
        labels = [e["label"] for e in result if e.get("type") == "demographic"]
        assert len(labels) == len(set(labels))

    def test_truncation(self):
        base = [{"type": "code", "label": f"C{i}", "weight": 1.0} for i in range(200)]
        result = prepend_demographic_tokens(base, target_vocab_size=50)
        assert len(result) == 50

    def test_demographics_include_sex_age_year(self):
        base = [{"type": "code", "label": f"C{i}", "weight": 1.0} for i in range(5)]
        target = len(base) + len(DEFAULT_DEMOGRAPHIC_LABELS)
        result = prepend_demographic_tokens(base, target_vocab_size=target)
        labels = {e["label"] for e in result if e.get("type") == "demographic"}
        assert "DEMO_SEX_MALE" in labels
        assert "DEMO_SEX_FEMALE" in labels
        assert "DEMO_AGE_0_4" in labels
        assert any(lbl.startswith("DEMO_YEAR_") for lbl in labels)
