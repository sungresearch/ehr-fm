"""Configuration model for vocabulary training."""

from pydantic import BaseModel


class VocabConfig(BaseModel):
    n_samples: int
    vocab_size: int
    n_numeric_bins: int = 10
    numeric_reservoir_size: int = 10000
    seed: int = None
    numeric_bin_by_unit: bool = False
    min_samples_per_unit: int | None = None
