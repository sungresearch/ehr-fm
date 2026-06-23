"""Immutable per-worker configuration for the pretokenize pipeline."""

import dataclasses
import datetime


@dataclasses.dataclass(frozen=True)
class PretokenizeWorkerConfig:
    """Immutable configuration for pretokenize worker processes.

    Built once in _pretokenize_init_worker and read by
    _pretokenize_process_row via a module-level global.
    """

    vocab_size: int
    age_mean: float
    age_std: float
    inject_time_intervals: bool
    interval_bins: list[tuple[float, float, str]]
    max_interval_repeat: int | None
    interval_token_lookup: dict[str, int]
    demographic_prefix: bool
    demographic_include_year: bool
    sex_codes_male: str | None
    sex_codes_female: str | None
    sex_codes_unknown: str | None

    def normalize_age(self, age: datetime.timedelta) -> float:
        """Normalize age using vocabulary statistics."""
        return (age.total_seconds() - self.age_mean) / self.age_std
