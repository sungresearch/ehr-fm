"""
Streaming statistics helpers for vocabulary construction.

Reservoir sampling and entropy-based weighting are adapted from FEMR
(https://github.com/som-shahlab/femr/tree/main)
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial

from torch.utils.data import DataLoader

from ehr_fm.logger import setup_logging
from ehr_fm.types import EventSequence


def compute_quantile_breaks(samples: list[float], num_quantiles: int) -> list[float]:
    """Compute deduplicated quantile break points for one code's samples.

    Sorts ``samples`` and splits them into ``num_quantiles`` equal buckets,
    returning the interior break points (length ``num_quantiles - 1`` before
    deduplication). Consecutive identical breaks are collapsed so that bucket
    boundaries are strictly increasing.

    Invariant inputs (all samples identical) return an empty list, which maps
    every value to the first bucket (Q:1). Callers are expected to skip codes
    with no samples; this helper assumes ``samples`` is non-empty.

    Returns:
        Sorted list of break points; ``[]`` for invariant samples.
    """
    samples = sorted(samples)

    # Invariant values (all identical) → empty breaks list → all map to Q:1
    if len(set(samples)) == 1:
        return []

    code_breaks = []
    for i in range(1, num_quantiles):
        idx = int(i * len(samples) / num_quantiles)
        if idx < len(samples):
            code_breaks.append(samples[idx])

    # Deduplicate consecutive identical breaks
    if not code_breaks:
        return []
    deduped = [code_breaks[0]]
    for brk in code_breaks[1:]:
        if brk != deduped[-1]:
            deduped.append(brk)
    return deduped


@dataclass(slots=True)
class OnlineStatistics:
    count: float = 0.0
    mean: float = 0.0
    M2: float = 0.0  # ∑ w*(x-mean)²

    def update(self, x: float, w: float = 1.0) -> None:
        self.count += w
        delta = x - self.mean
        self.mean += delta * (w / self.count)
        self.M2 += w * delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass(slots=True)
class ReservoirSampler:
    size: int
    seed: int | None = None
    samples: list[float] = field(init=False)
    total_weight: float = 0.0
    n: int = 0
    j: float = 0.0
    p_none: float = 1.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self.samples = [math.nan] * self.size
        self._rng = random.Random(self.seed)

    def update(self, x: float, w: float = 1.0) -> None:
        self.total_weight += w
        if self.n < self.size:  # fill first
            self.samples[self.n] = x
            self.n += 1
            if self.n == self.size:
                self.j = self._rng.random()
                self.p_none = 1.0
            return

        prob = w / self.total_weight
        self.j -= prob * self.p_none
        self.p_none *= 1.0 - prob
        if self.j <= 0.0:
            idx = self._rng.randrange(self.size)
            self.samples[idx] = x
            self.j = self._rng.random()
            self.p_none = 1.0


# =============================================================================
# Quantile Pre-Scanner for Joint/Factorized Vocabulary Training
# =============================================================================


class QuantilePreScanner:
    """Pre-scan data to compute stable quantile breaks before vocabulary training.

    This class performs the first pass of two-pass vocabulary training:
    1. Pre-scan: Collect numeric samples and compute quantile breaks (this class)
    2. Main training: Use fixed breaks to build vocabulary entries

    This ensures consistent quantile assignment across all patients during
    vocabulary training, which is critical for joint mode where combined
    tokens like "glucose/Q:3" need stable bucket assignments.

    Args:
        num_quantiles: Number of quantile buckets (default 10)
        reservoir_size: Max samples per code for break computation
        seed: Random seed for reservoir sampling
        numeric_bin_by_unit: Whether to track numeric values by (code, unit) pair
    """

    def __init__(
        self,
        num_quantiles: int = 10,
        reservoir_size: int = 10_000,
        seed: int | None = None,
        numeric_bin_by_unit: bool = False,
    ):
        self.num_quantiles = num_quantiles
        self.reservoir_size = reservoir_size
        self.seed = seed
        self.numeric_bin_by_unit = numeric_bin_by_unit
        self.logger = setup_logging(child_name=self.__class__.__name__)

        # Reservoirs for collecting numeric samples
        self.reservoirs: dict[str, ReservoirSampler] = defaultdict(
            partial(ReservoirSampler, size=reservoir_size, seed=seed)
        )

        # Also discover stages during pre-scan
        self.discovered_stages: set[str] = set()

    def forward(self, batch: list[EventSequence], ids: list[int] = None) -> None:
        """Process a batch of event sequences.

        Collects numeric values into reservoirs and discovers workflow stages.

        Args:
            batch: List of event sequences (each sequence is a list of event dicts)
            ids: Optional sample IDs for logging
        """
        for i, events in enumerate(batch):
            if len(events) == 0:
                continue

            for event in events:
                code = event["code"]
                numeric_value = event["numeric_value"]
                workflow_stage = event.get("workflow_stage")

                # Collect numeric samples for quantile break computation
                if numeric_value is not None:
                    if self.numeric_bin_by_unit:
                        unit = event.get("unit")
                        reservoir_key = f"{code}|{unit}" if unit else code
                    else:
                        reservoir_key = code
                    self.reservoirs[reservoir_key].update(numeric_value)

                # Discover workflow stages
                if workflow_stage is not None:
                    self.discovered_stages.add(workflow_stage.lower())

    def scan(self, dataloader: DataLoader) -> None:
        """Scan through all data to collect samples and compute breaks.

        Args:
            dataloader: DataLoader to iterate through
        """
        self.logger.info("Pre-scanning data for quantile breaks...")
        n_batches = len(dataloader)
        scan_start = time.time()

        for i, batch in enumerate(dataloader):
            _, sequences = batch
            self.forward(sequences)

            if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                elapsed = time.time() - scan_start
                self.logger.info(f"Pre-scan batch {i + 1}/{n_batches} ({elapsed:.1f}s elapsed)")

        elapsed = time.time() - scan_start
        self.logger.info(
            f"Pre-scan complete: {len(self.reservoirs)} codes with numeric values, "
            f"{len(self.discovered_stages)} stages discovered ({elapsed:.1f}s)"
        )

    def compute_breaks(self) -> dict[str, list[float]]:
        """Compute quantile breaks from collected samples.

        For each code with numeric values, computes break points that divide
        the samples into num_quantiles equal buckets.

        Handles invariant values (all samples identical) by returning empty
        breaks list, which maps all values to bucket 1.

        Returns:
            Dict mapping code → [break1, break2, ...] where len = num_quantiles - 1
        """
        breaks = {}

        for code, reservoir in self.reservoirs.items():
            if reservoir.n == 0:
                continue

            samples = reservoir.samples[: reservoir.n]
            if not samples:
                continue

            breaks[code] = compute_quantile_breaks(samples, self.num_quantiles)

        self.logger.info(f"Computed quantile breaks for {len(breaks)} codes")
        return breaks

    def get_discovered_stages(self) -> list[str]:
        """Get list of discovered workflow stages.

        Returns:
            Sorted list of discovered stage values (lowercase)
        """
        return sorted(self.discovered_stages)
