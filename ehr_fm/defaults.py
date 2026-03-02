"""
Centralized defaults for ehr_fm.

- Interval token definitions (labels and bin edges)
- Demographic token labels (sex, WHO 5-year age buckets, 5-year year buckets)
- Sex code defaults

These defaults are intentionally hard-coded to ensure stable, reproducible behavior.
They are referenced by both the vocabulary builder and the tokenizer/pretokenizer.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Final


def _seconds(*, minutes: float = 0.0, hours: float = 0.0, days: float = 0.0, months: float = 0.0) -> float:
    """Convert composite time units to seconds. Month interpreted as 30 days."""

    minute = 60.0
    hour = 60.0 * minute
    day = 24.0 * hour
    month = 30.0 * day
    return minutes * minute + hours * hour + days * day + months * month


# Interval bins [min, max) with stable labels. Month=30 days.
# Minimum interval: 5m. Bins: 5m-15m, 15m-1h, then hourly/daily/monthly progression.
DEFAULT_INTERVAL_BINS: Final[list[tuple[float, float, str]]] = [
    (_seconds(minutes=5), _seconds(minutes=15), "INT_5m_15m"),
    (_seconds(minutes=15), _seconds(hours=1), "INT_15m_1h"),
    (_seconds(hours=1), _seconds(hours=2), "INT_1h_2h"),
    (_seconds(hours=2), _seconds(hours=6), "INT_2h_6h"),
    (_seconds(hours=6), _seconds(hours=12), "INT_6h_12h"),
    (_seconds(hours=12), _seconds(days=1), "INT_12h_1d"),
    (_seconds(days=1), _seconds(days=3), "INT_1d_3d"),
    (_seconds(days=3), _seconds(days=7), "INT_3d_1w"),
    (_seconds(days=7), _seconds(days=14), "INT_1w_2w"),
    (_seconds(days=14), _seconds(months=1), "INT_2w_1mt"),
    (_seconds(months=1), _seconds(months=3), "INT_1mt_3mt"),
    (_seconds(months=3), _seconds(months=6), "INT_3mt_6mt"),
]

# Repeatable 6-month interval label (emitted as repeats, not a ranged bin)
REPEATABLE_INTERVAL_LABEL: Final[str] = "INT_6mt"

DEFAULT_INTERVAL_LABELS: Final[list[str]] = [label for _, _, label in DEFAULT_INTERVAL_BINS] + [
    REPEATABLE_INTERVAL_LABEL
]


# Demographic defaults
DEMO_YEAR_MIN: Final[int] = 1970
DEMO_YEAR_MAX: Final[int] = 2079


def _iter_age_bucket_labels() -> Iterable[str]:
    for start in range(0, 100, 5):
        end = start + 4
        yield f"DEMO_AGE_{start}_{end}"
    yield "DEMO_AGE_100P"


def _iter_year_bucket_labels() -> Iterable[str]:
    for year in range(DEMO_YEAR_MIN, DEMO_YEAR_MAX + 1, 5):
        yield f"DEMO_YEAR_{year}_{year + 4}"


DEFAULT_DEMOGRAPHIC_LABELS: Final[list[str]] = (
    [
        # Sex
        "DEMO_SEX_MALE",
        "DEMO_SEX_FEMALE",
        "DEMO_SEX_UNKNOWN",
        # Ages
    ]
    + list(_iter_age_bucket_labels())
    + list(_iter_year_bucket_labels())
)


# Sex code defaults (overridable via CLI in pretokenize)
DEFAULT_SEX_CODES: Final[dict[str, set[str]]] = {
    "male": {"8507"},
    "female": {"8532"},
    "unknown": set(),
}


# =============================================================================
# Factorized tokenization special tokens
# =============================================================================
# Only UNK tokens predefined; value tokens discovered at training time

QUANTILE_UNK_LABEL: Final[str] = "Q:UNK"
STAGE_UNK_LABEL: Final[str] = "STAGE:UNK"


def quantile_labels(num_quantiles: int = 10) -> list[str]:
    """Generate quantile token labels Q:1 through Q:n plus Q:UNK.

    Args:
        num_quantiles: Number of buckets (default 10)

    Returns:
        List of labels: ["Q:1", "Q:2", ..., "Q:n", "Q:UNK"]
    """
    return [f"Q:{i}" for i in range(1, num_quantiles + 1)] + [QUANTILE_UNK_LABEL]
