"""Tests for compute_numeric_stats.compute_log_stats (per-code log z-score stats)."""

import math

import pytest

from ehr_fm.scripts.compute_numeric_stats import compute_log_stats


def test_mean_and_sample_std():
    vals = [1.0, 2.0, 3.0, 4.0]  # mean 2.5, sample variance 5/3
    out = compute_log_stats({"LAB/x": vals})["LAB/x"]
    assert out["n_samples"] == 4
    assert out["log_mean"] == pytest.approx(2.5)
    assert out["log_std"] == pytest.approx(math.sqrt(5.0 / 3.0))  # divides by n-1


def test_single_value_std_fallback():
    out = compute_log_stats({"LAB/x": [3.0]})["LAB/x"]
    assert out["n_samples"] == 1
    assert out["log_mean"] == 3.0
    assert out["log_std"] == 1.0  # n <= 1 -> std forced to 1.0


def test_degenerate_spread_std_fallback():
    # All identical -> variance 0 (< 1e-8) -> std forced to 1.0.
    out = compute_log_stats({"LAB/x": [5.0, 5.0, 5.0]})["LAB/x"]
    assert out["log_mean"] == 5.0
    assert out["log_std"] == 1.0


def test_empty_code_omitted():
    assert compute_log_stats({"LAB/x": []}) == {}


def test_multiple_codes_independent():
    out = compute_log_stats({"A": [1.0, 3.0], "B": [10.0]})
    assert set(out) == {"A", "B"}
    assert out["A"]["log_mean"] == pytest.approx(2.0)
    assert out["A"]["log_std"] == pytest.approx(math.sqrt(2.0))  # var = ((1-2)^2+(3-2)^2)/1 = 2
    assert out["B"]["log_std"] == 1.0  # single sample
