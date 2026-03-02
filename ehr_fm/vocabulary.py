"""
Reservoir sampling and entropy-based weighting are adapted from FEMR
(https://github.com/som-shahlab/femr/tree/main)
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from torch.utils.data import DataLoader

from .defaults import (
    DEFAULT_DEMOGRAPHIC_LABELS,
    DEFAULT_INTERVAL_LABELS,
    STAGE_UNK_LABEL,
    quantile_labels,
)
from .io import write_dict_to_json
from .logger import setup_logging
from .types import ConfigLike, EventSequence, PathLike
from .validation import PathValidator, VocabConfig, validate_config


def _default_interval_vocab_entries() -> list[dict[str, Any]]:
    """Build interval token entries with stable labels and type 'interval'.

    Order matters: these are prepended to ensure low IDs and inclusion.
    Includes repeatable 6-month token label at the end.
    """
    return [{"type": "interval", "label": lbl, "weight": -1.0} for lbl in DEFAULT_INTERVAL_LABELS]


def prepend_interval_tokens(
    vocab_entries: list[dict[str, Any]], target_vocab_size: int | None = None
) -> list[dict[str, Any]]:
    """Prepend interval tokens to the ranked vocabulary, then truncate to original size.

    - Ensures interval tokens receive the lowest IDs and are never truncated out.
    - Avoids duplicates if intervals already exist.
    """
    intervals = _default_interval_vocab_entries()
    existing_labels = {v.get("label") for v in vocab_entries if v.get("type") == "interval"}
    to_add = [e for e in intervals if e["label"] not in existing_labels]

    combined = to_add + vocab_entries
    # Preserve or cap at configured vocab size if provided
    target_size = target_vocab_size if target_vocab_size is not None else len(vocab_entries)
    if len(combined) > target_size:
        return combined[:target_size]
    return combined


def _default_demographic_vocab_entries() -> list[dict[str, Any]]:
    """Build demographic token entries: sex, age buckets, first-event year buckets."""
    return [{"type": "demographic", "label": lbl, "weight": -1.0} for lbl in DEFAULT_DEMOGRAPHIC_LABELS]


def prepend_demographic_tokens(
    vocab_entries: list[dict[str, Any]], target_vocab_size: int | None = None
) -> list[dict[str, Any]]:
    """Prepend demographic tokens; cap to target_vocab_size if provided."""
    demos = _default_demographic_vocab_entries()
    existing_labels = {v.get("label") for v in vocab_entries if v.get("type") == "demographic"}
    to_add = [e for e in demos if e["label"] not in existing_labels]
    combined = to_add + vocab_entries
    target_size = target_vocab_size if target_vocab_size is not None else len(vocab_entries)
    if len(combined) > target_size:
        return combined[:target_size]
    return combined


# =============================================================================
# Factorized tokenization vocabulary helpers
# =============================================================================


def _factorized_quantile_vocab_entries(num_quantiles: int = 10) -> list[dict[str, Any]]:
    """Build quantile token entries Q:1...Q:n and Q:UNK.

    These are prepended to vocabulary to ensure stable low IDs.

    Args:
        num_quantiles: Number of quantile buckets

    Returns:
        List of vocab entries with type 'quantile'
    """
    return [{"type": "quantile", "label": lbl, "weight": -1.0} for lbl in quantile_labels(num_quantiles)]


def _factorized_stage_vocab_entries(discovered_stages: list[str]) -> list[dict[str, Any]]:
    """Build stage token entries from discovered stages plus STAGE:UNK.

    Args:
        discovered_stages: Stage values found during training

    Returns:
        List of vocab entries for STAGE:* tokens
    """
    entries = [
        {"type": "stage", "label": f"STAGE:{s.lower()}", "weight": -1.0}
        for s in sorted(set(discovered_stages))
    ]
    # Always include UNK
    entries.append({"type": "stage", "label": STAGE_UNK_LABEL, "weight": -1.0})
    return entries


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

            samples = sorted(reservoir.samples[: reservoir.n])
            if not samples:
                continue

            # Handle invariant values (all samples identical)
            unique_values = set(samples)
            if len(unique_values) == 1:
                # All values are the same → empty breaks list
                # All values will map to Q:1
                breaks[code] = []
                self.logger.debug(f"Code '{code}' has invariant value: {samples[0]}")
                continue

            # Compute quantile break points
            code_breaks = []
            for i in range(1, self.num_quantiles):
                idx = int(i * len(samples) / self.num_quantiles)
                if idx < len(samples):
                    code_breaks.append(samples[idx])

            # Deduplicate consecutive identical breaks
            if code_breaks:
                deduped = [code_breaks[0]]
                for brk in code_breaks[1:]:
                    if brk != deduped[-1]:
                        deduped.append(brk)
                code_breaks = deduped

            breaks[code] = code_breaks

        self.logger.info(f"Computed quantile breaks for {len(breaks)} codes")
        return breaks

    def get_discovered_stages(self) -> list[str]:
        """Get list of discovered workflow stages.

        Returns:
            Sorted list of discovered stage values (lowercase)
        """
        return sorted(self.discovered_stages)


class Vocab:
    def __init__(self, config: ConfigLike):
        # configs
        self.config = validate_config(config, VocabConfig)
        self.logger = setup_logging(child_name=self.__class__.__name__)
        self.n_samples = self.config.n_samples
        self.vocab_size = self.config.vocab_size
        self.n_numeric_bins = self.config.n_numeric_bins
        self.numeric_reservoir_size = self.config.numeric_reservoir_size
        self.numeric_bin_by_unit = self.config.numeric_bin_by_unit
        self.min_samples_per_unit = self.config.min_samples_per_unit

        # stats trackers
        self.age_stats = OnlineStatistics()
        self.numeric_reservoirs: dict[tuple[str, int | str | None], ReservoirSampler] = defaultdict(
            partial(
                ReservoirSampler,
                size=self.numeric_reservoir_size,
                seed=self.config.seed,
            )
        )
        self.code_weights: dict[str, float] = defaultdict(float)
        self.text_weights: dict[tuple[str, str], float] = defaultdict(float)

    def forward(self, batch: list[EventSequence], ids: list[int] = None) -> None:
        for i, events in enumerate(batch):
            total_events = len(events)
            if total_events == 0:
                warning = f"Found no events for sample #{i} in current batch."
                if ids:
                    warning += f" Sample ID: '{ids[i]}'."
                self.logger.warning(warning)
                continue

            birth_date = next((e["time"] for e in events if e["code"] == "MEDS_BIRTH"), None)

            weight = 1.0 / (self.n_samples * total_events)
            reservoir_calls: list[tuple[ReservoirSampler, float]] = []
            for event in events:
                code = event["code"]
                numeric_value = event["numeric_value"]
                text_value = event["text_value"]
                t = event["time"]
                if t != birth_date:
                    self.age_stats.update((t - birth_date).total_seconds(), weight)

                if numeric_value is not None:
                    unit = event.get("unit") if self.numeric_bin_by_unit else None
                    reservoir_key = (code, unit)
                    reservoir_calls.append((self.numeric_reservoirs[reservoir_key], numeric_value))
                elif text_value is not None:
                    self.text_weights[(code, text_value)] += weight
                else:
                    self.code_weights[code] += weight

            for reservoir, numeric_value in reservoir_calls:
                reservoir.update(numeric_value, weight)

    def train(self, dataloader: DataLoader) -> None:
        self.logger.info("Training vocab")
        n_batches = len(dataloader)

        # Track overall training start time
        training_start = time.time()
        batch_start = training_start

        for i, batch in enumerate(dataloader):
            # Time since training started
            current_time = time.time()
            time_since_start = current_time - training_start
            time_since_last_batch = current_time - batch_start

            # Process the batch
            processing_start = time.time()
            _, sequences = batch
            self.forward(sequences)
            processing_end = time.time()

            # Calculate timing metrics
            processing_time = processing_end - processing_start
            batch_overhead = time_since_last_batch - processing_time if i > 0 else 0.0

            # Log comprehensive timing information
            if i == 0:
                self.logger.info(
                    f"Batch {i + 1}/{n_batches}: processed in {processing_time:.4f}s "
                    f"(cumulative: {time_since_start:.1f}s)"
                )
            else:
                self.logger.info(
                    f"Batch {i + 1}/{n_batches}: processed in {processing_time:.4f}s, "
                    f"overhead: {batch_overhead:.4f}s, total batch time: {time_since_last_batch:.4f}s "
                    f"(cumulative: {time_since_start:.1f}s)"
                )

            # Update batch start time for next iteration
            batch_start = time.time()

    def get_vocab(self) -> list[dict[str, Any]]:
        vocab = []
        for code, weight in self.code_weights.items():
            entry = {
                "type": "code",
                "code_string": code,
                "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
            }
            vocab.append(entry)

        for (code, text), weight in self.text_weights.items():
            entry = {
                "type": "text",
                "code_string": code,
                "text_string": text,
                "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
            }
            vocab.append(entry)

        for (code, unit), reservoir in self.numeric_reservoirs.items():
            # Apply min_samples_per_unit filter
            if self.min_samples_per_unit is not None and reservoir.n < self.min_samples_per_unit:
                continue

            weight = reservoir.total_weight / self.n_numeric_bins
            n_samples = reservoir.n
            samples = sorted(reservoir.samples[: reservoir.n])  # drops nans

            samples_per_bin = (n_samples + self.n_numeric_bins - 1) // self.n_numeric_bins

            bins_created = 0
            for bin_index in range(self.n_numeric_bins):
                if bin_index == 0:
                    start_val = -math.inf
                else:
                    if bin_index * samples_per_bin >= n_samples:
                        continue
                    start_val = samples[bin_index * samples_per_bin]

                if (bin_index == self.n_numeric_bins - 1) or ((bin_index + 1) * samples_per_bin >= n_samples):
                    end_val = math.inf
                else:
                    end_val = samples[(bin_index + 1) * samples_per_bin]

                # Skip bins where start == end (constant values within bin)
                if start_val == end_val:
                    continue

                entry = {
                    "type": "numeric",
                    "code_string": code,
                    "unit": unit,
                    "val_start": start_val,
                    "val_end": end_val,
                    "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
                }
                vocab.append(entry)
                bins_created += 1

            # If all bins were skipped due to constant values, create a single (-inf, inf) bin
            # This ensures codes with constant numeric values still get a vocabulary entry
            if bins_created == 0 and n_samples > 0:
                entry = {
                    "type": "numeric",
                    "code_string": code,
                    "unit": unit,
                    "val_start": -math.inf,
                    "val_end": math.inf,
                    "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
                }
                vocab.append(entry)

        vocab.sort(key=lambda a: a["weight"])
        vocab = vocab[: self.vocab_size]
        return vocab

    def save(
        self,
        output_path: PathLike,
        save_reservoirs: bool = False,
        overwrite: bool = False,
        precomputed_vocab: list[dict[str, Any]] | None = None,
    ) -> None:
        output_path = PathValidator(path=output_path).path

        output_path.mkdir(exist_ok=True, parents=True)
        vocab_entries = precomputed_vocab if precomputed_vocab is not None else self.get_vocab()
        result = {
            "vocab": vocab_entries,
            "age_stats": {
                "mean": self.age_stats.mean,
                "std": self.age_stats.std,
            },
            "config": {
                "numeric_bin_by_unit": self.numeric_bin_by_unit,
                "min_samples_per_unit": self.min_samples_per_unit,
            },
        }

        write_dict_to_json(data=result, path=output_path / "vocab.json", overwrite=overwrite)

        if save_reservoirs:
            reservoirs = {
                "code_weights": dict(self.code_weights),
                "numeric_reservoirs": {
                    f"{code}|{unit if unit else 'NULL'}": {
                        "total_weight": sampler.total_weight,
                        "samples": sampler.samples[: sampler.n],
                    }
                    for (code, unit), sampler in self.numeric_reservoirs.items()
                },
                "text_weights": dict(self.text_weights),
            }

            write_dict_to_json(data=reservoirs, path=output_path / "reservoirs.json", overwrite=overwrite)


# =============================================================================
# Factorized Vocabulary Classes
# =============================================================================


class FactorizedVocab(Vocab):
    """Factorized vocabulary builder for on-the-fly attribute tokenization.

    Unlike base Vocab which creates code-specific numeric/text entries,
    FactorizedVocab creates:
    - Generic Q:1...Q:n tokens for quantile buckets
    - Generic STAGE:* tokens for discovered workflow stages
    - Generic TXT:* tokens for discovered text values
    - Base code tokens (with optional prefix removal)

    This enables on-the-fly attribute emission during pretokenization
    using the FactorizedPolicy.

    Supports two training modes:
    1. Single-pass (default): Collects samples and computes breaks after training
    2. Two-pass: Uses pre-computed breaks from QuantilePreScanner

    Args:
        config: VocabConfig with vocab_size, n_numeric_bins, etc.
        num_quantiles: Number of quantile buckets (default 10)
        emit_quantiles: Whether to emit Q:* tokens
        emit_text: Whether to emit TXT:* tokens
        emit_stage: Whether to emit STAGE:* tokens
        remove_prefixes: Whether to remove code prefixes (e.g., LAB/glucose → glucose)
        separator: Separator character for prefix removal
        precomputed_quantile_breaks: Pre-computed breaks from QuantilePreScanner (optional)
        precomputed_stages: Pre-discovered stages from QuantilePreScanner (optional)
    """

    def __init__(
        self,
        config: ConfigLike,
        num_quantiles: int = 10,
        emit_quantiles: bool = True,
        emit_text: bool = True,
        emit_stage: bool = True,
        remove_prefixes: bool = True,
        separator: str = "/",
        precomputed_quantile_breaks: dict[str, list[float]] | None = None,
        precomputed_stages: list[str] | None = None,
    ):
        super().__init__(config)
        self.num_quantiles = num_quantiles
        self.emit_quantiles = emit_quantiles
        self.emit_text = emit_text
        self.emit_stage = emit_stage
        self.remove_prefixes = remove_prefixes
        self.separator = separator

        # Pre-computed quantile breaks (from two-pass training)
        self._precomputed_quantile_breaks = precomputed_quantile_breaks

        # Additional tracking for factorized mode
        self.discovered_stages: set[str] = set(precomputed_stages or [])
        self.base_code_weights: dict[str, float] = defaultdict(float)
        self.text_value_weights: dict[str, float] = defaultdict(float)

    def _extract_base_code(self, code: str) -> str:
        """Extract base code from full code string."""
        if not self.remove_prefixes:
            return code
        if self.separator not in code:
            return code
        parts = code.split(self.separator)
        if len(parts) > 1 and parts[1]:
            return parts[1]
        return code

    def _normalize_text(self, text: str) -> str | None:
        """Normalize text value for tokenization."""
        import re

        if not text:
            return None
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", "_", normalized)
        normalized = re.sub(r"[^\w_]", "", normalized)
        return normalized if normalized else None

    def forward(self, batch: list[EventSequence], ids: list[int] = None) -> None:
        """Process batch and collect statistics for factorized vocabulary.

        If pre-computed quantile breaks were provided, skips reservoir collection.
        """
        # Skip reservoir collection if breaks were pre-computed
        skip_reservoir = self._precomputed_quantile_breaks is not None

        for i, events in enumerate(batch):
            total_events = len(events)
            if total_events == 0:
                warning = f"Found no events for sample #{i} in current batch."
                if ids:
                    warning += f" Sample ID: '{ids[i]}'."
                self.logger.warning(warning)
                continue

            birth_date = next((e["time"] for e in events if e["code"] == "MEDS_BIRTH"), None)

            weight = 1.0 / (self.n_samples * total_events)
            reservoir_calls: list[tuple[ReservoirSampler, float]] = []

            for event in events:
                code = event["code"]
                numeric_value = event["numeric_value"]
                text_value = event["text_value"]
                workflow_stage = event.get("workflow_stage")
                t = event["time"]

                if t != birth_date:
                    self.age_stats.update((t - birth_date).total_seconds(), weight)

                # Extract base code and accumulate weight
                base_code = self._extract_base_code(code)
                self.base_code_weights[base_code] += weight

                # Collect numeric values for quantile break computation (if not pre-computed)
                if not skip_reservoir and numeric_value is not None and self.emit_quantiles:
                    unit = event.get("unit") if self.numeric_bin_by_unit else None
                    reservoir_key = (code, unit)  # Use full code for breaks lookup
                    reservoir_calls.append((self.numeric_reservoirs[reservoir_key], numeric_value))

                # Collect text values
                if text_value is not None and self.emit_text:
                    normalized = self._normalize_text(text_value)
                    if normalized:
                        self.text_value_weights[normalized] += weight

                # Discover stage values (if not pre-computed)
                if workflow_stage is not None and self.emit_stage:
                    self.discovered_stages.add(workflow_stage.lower())

            if not skip_reservoir:
                for reservoir, numeric_value in reservoir_calls:
                    reservoir.update(numeric_value, weight)

    def get_quantile_breaks(self) -> dict[str, list[float]]:
        """Get quantile breaks for each code.

        If pre-computed breaks were provided (two-pass training), returns those.
        Otherwise, computes breaks from collected samples (single-pass training).

        Returns:
            Dict mapping code → [break1, break2, ...] where len = num_quantiles - 1
        """
        # Use pre-computed breaks if available
        if self._precomputed_quantile_breaks is not None:
            return self._precomputed_quantile_breaks

        # Compute from reservoirs (single-pass mode)
        breaks = {}
        for (code, unit), reservoir in self.numeric_reservoirs.items():
            if reservoir.n == 0:
                continue

            samples = sorted(reservoir.samples[: reservoir.n])
            if not samples:
                continue

            # Handle invariant values (all samples identical)
            unique_values = set(samples)
            if len(unique_values) == 1:
                # All values are the same → empty breaks list → all map to Q:1
                breaks[code] = []
                continue

            # Compute break points for num_quantiles buckets
            code_breaks = []
            for i in range(1, self.num_quantiles):
                idx = int(i * len(samples) / self.num_quantiles)
                if idx < len(samples):
                    code_breaks.append(samples[idx])

            # Deduplicate consecutive identical breaks
            if code_breaks:
                deduped = [code_breaks[0]]
                for brk in code_breaks[1:]:
                    if brk != deduped[-1]:
                        deduped.append(brk)
                code_breaks = deduped

            breaks[code] = code_breaks

        return breaks

    def get_vocab(self) -> tuple[list[dict[str, Any]], int]:
        """Generate factorized vocabulary with generic attribute tokens.

        Returns vocab entries in order:
        1. Q:1, Q:2, ..., Q:n, Q:UNK (quantile tokens)
        2. STAGE:*, STAGE:UNK (stage tokens)
        3. TXT:* (text tokens, sorted by weight)
        4. Base code tokens (sorted by weight)

        Returns:
            Tuple of (vocab_entries, num_ntp_classes) where:
            - vocab_entries: List of vocab entries
            - num_ntp_classes: Number of tokens that are NTP targets
              (= Q + S + T + event codes for factorized)
        """
        vocab = []

        # 1. Quantile tokens (always prepended for stable IDs)
        if self.emit_quantiles:
            vocab.extend(_factorized_quantile_vocab_entries(self.num_quantiles))

        # 2. Stage tokens (discovered + UNK)
        if self.emit_stage:
            vocab.extend(_factorized_stage_vocab_entries(list(self.discovered_stages)))

        # 3. Text tokens (sorted by weight)
        text_entries = []
        for normalized_text, weight in self.text_value_weights.items():
            if weight >= 1.0:
                weight = 1.0 - 1e-8
            if weight <= 0.0:
                weight = 1e-8
            entry = {
                "type": "text",
                "label": f"TXT:{normalized_text}",
                "text_string": normalized_text,
                "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
            }
            text_entries.append(entry)

        text_entries.sort(key=lambda x: x["weight"])
        vocab.extend(text_entries)

        # 4. Base code tokens (sorted by weight)
        code_entries = []
        for code, weight in self.base_code_weights.items():
            if weight >= 1.0:
                weight = 1.0 - 1e-8
            if weight <= 0.0:
                weight = 1e-8
            entry = {
                "type": "code",
                "code_string": code,
                "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
            }
            code_entries.append(entry)

        code_entries.sort(key=lambda x: x["weight"])
        vocab.extend(code_entries)

        # Calculate num_ntp_classes
        # For factorized: all tokens up to this point are NTP targets
        num_ntp_classes = len(vocab)

        # Truncate to vocab_size
        vocab = vocab[: self.vocab_size]

        return vocab, num_ntp_classes

    def save(
        self,
        output_path: PathLike,
        save_reservoirs: bool = False,
        overwrite: bool = False,
        precomputed_vocab: tuple[list[dict[str, Any]], int] | None = None,
    ) -> None:
        """Save factorized vocabulary with quantile breaks."""
        output_path = PathValidator(path=output_path).path
        output_path.mkdir(exist_ok=True, parents=True)

        if precomputed_vocab is not None:
            vocab_entries, num_ntp_classes = precomputed_vocab
        else:
            vocab_entries, num_ntp_classes = self.get_vocab()

        result = {
            "vocab": vocab_entries,
            "age_stats": {
                "mean": self.age_stats.mean,
                "std": self.age_stats.std,
            },
            "config": {
                "numeric_bin_by_unit": self.numeric_bin_by_unit,
                "min_samples_per_unit": self.min_samples_per_unit,
                "tokenization_mode": "factorized",
                "num_ntp_classes": num_ntp_classes,
                "num_parent_codes": 0,
                "num_quantiles": self.num_quantiles,
                "emit_quantiles": self.emit_quantiles,
                "emit_text": self.emit_text,
                "emit_stage": self.emit_stage,
                "remove_prefixes": self.remove_prefixes,
                "separator": self.separator,
            },
            "quantile_breaks": self.get_quantile_breaks(),
            "discovered_stages": sorted(self.discovered_stages),
        }

        write_dict_to_json(data=result, path=output_path / "vocab.json", overwrite=overwrite)

        if save_reservoirs:
            reservoirs = {
                "base_code_weights": dict(self.base_code_weights),
                "text_value_weights": dict(self.text_value_weights),
                "numeric_reservoirs": {
                    f"{code}|{unit if unit else 'NULL'}": {
                        "total_weight": sampler.total_weight,
                        "samples": sampler.samples[: sampler.n],
                    }
                    for (code, unit), sampler in self.numeric_reservoirs.items()
                },
                "discovered_stages": sorted(self.discovered_stages),
                "parent_codes": [],
            }
            write_dict_to_json(data=reservoirs, path=output_path / "reservoirs.json", overwrite=overwrite)


# =============================================================================
# Joint Vocabulary Class
# =============================================================================


class JointVocab:
    """Joint vocabulary builder with combined token entries.

    Creates vocabulary entries for combined tokens like "glucose/Q:3" that
    concatenate base codes with attribute tokens. Uses pre-computed quantile
    breaks from QuantilePreScanner for stable bucket assignment.

    Unlike FactorizedVocab which creates separate entries for base codes and
    attribute tokens, JointVocab creates a single entry per unique combination.

    Requires two-pass training:
    1. Pre-scan (QuantilePreScanner): Compute quantile breaks
    2. Main training (this class): Count combined token frequencies

    Args:
        config: VocabConfig with vocab_size, n_numeric_bins, etc.
        quantile_breaks: Pre-computed {code: [breaks]} from QuantilePreScanner
        discovered_stages: Pre-discovered stage values from QuantilePreScanner
        num_quantiles: Number of quantile buckets
        emit_quantiles: Whether to append Q:* for numeric values
        emit_text: Whether to append TXT:* for text values
        emit_stage: Whether to append STAGE:* for workflow_stage
        remove_prefixes: Whether to strip code prefixes
        separator: Token part separator
    """

    def __init__(
        self,
        config: ConfigLike,
        quantile_breaks: dict[str, list[float]],
        discovered_stages: list[str] | None = None,
        num_quantiles: int = 10,
        emit_quantiles: bool = True,
        emit_text: bool = True,
        emit_stage: bool = True,
        remove_prefixes: bool = True,
        separator: str = "/",
    ):
        from .tokenization.joint import JointConfig, JointTokenBuilder

        self.config = validate_config(config, VocabConfig)
        self.logger = setup_logging(child_name=self.__class__.__name__)
        self.n_samples = self.config.n_samples
        self.vocab_size = self.config.vocab_size

        # Store quantile breaks from pre-scan
        self.quantile_breaks = quantile_breaks
        self.discovered_stages = set(discovered_stages or [])
        self.num_quantiles = num_quantiles
        self.emit_quantiles = emit_quantiles
        self.emit_text = emit_text
        self.emit_stage = emit_stage
        self.remove_prefixes = remove_prefixes
        self.separator = separator

        # Create JointTokenBuilder for consistent token generation
        joint_config = JointConfig(
            emit_quantiles=emit_quantiles,
            emit_text=emit_text,
            emit_stage=emit_stage,
            num_quantiles=num_quantiles,
            remove_prefixes=remove_prefixes,
            separator=separator,
        )
        self.token_builder = JointTokenBuilder(
            config=joint_config,
            quantile_breaks=quantile_breaks,
            known_stages=self.discovered_stages,
        )

        # Stats trackers
        self.age_stats = OnlineStatistics()
        self.combined_weights: dict[str, float] = defaultdict(float)

    def forward(self, batch: list[EventSequence], ids: list[int] = None) -> None:
        """Process batch and accumulate combined token weights.

        Uses JointTokenBuilder to create combined tokens and accumulates
        frequency weights for each unique token string.

        Args:
            batch: List of event sequences
            ids: Optional sample IDs for logging
        """
        for i, events in enumerate(batch):
            total_events = len(events)
            if total_events == 0:
                warning = f"Found no events for sample #{i} in current batch."
                if ids:
                    warning += f" Sample ID: '{ids[i]}'."
                self.logger.warning(warning)
                continue

            birth_date = next((e["time"] for e in events if e["code"] == "MEDS_BIRTH"), None)

            weight = 1.0 / (self.n_samples * total_events)

            for event in events:
                t = event["time"]
                if t != birth_date:
                    self.age_stats.update((t - birth_date).total_seconds(), weight)

                # Build combined token and accumulate weight
                combined_token = self.token_builder.build_token_from_event(event)
                if combined_token:
                    self.combined_weights[combined_token] += weight

    def train(self, dataloader: DataLoader) -> None:
        """Train vocabulary by accumulating combined token frequencies.

        Args:
            dataloader: DataLoader to iterate through
        """
        mode = "joint"
        self.logger.info(f"Training {mode} vocabulary...")
        n_batches = len(dataloader)
        training_start = time.time()
        batch_start = training_start

        for i, batch in enumerate(dataloader):
            current_time = time.time()
            time_since_start = current_time - training_start
            time_since_last_batch = current_time - batch_start

            processing_start = time.time()
            _, sequences = batch
            self.forward(sequences)
            processing_end = time.time()

            processing_time = processing_end - processing_start
            batch_overhead = time_since_last_batch - processing_time if i > 0 else 0.0

            if i == 0:
                self.logger.info(
                    f"Batch {i + 1}/{n_batches}: processed in {processing_time:.4f}s "
                    f"(cumulative: {time_since_start:.1f}s)"
                )
            else:
                self.logger.info(
                    f"Batch {i + 1}/{n_batches}: processed in {processing_time:.4f}s, "
                    f"overhead: {batch_overhead:.4f}s, total batch time: {time_since_last_batch:.4f}s "
                    f"(cumulative: {time_since_start:.1f}s)"
                )

            batch_start = time.time()

        self.logger.info(f"Joint vocabulary training complete: {len(self.combined_weights)} unique tokens")

    def get_vocab(self) -> tuple[list[dict[str, Any]], int]:
        """Generate joint vocabulary with combined token entries.

        All event entries have type "code" since each represents a single token
        that combines base code with attribute parts.

        Returns:
            Tuple of (vocab_entries, num_event_codes) where:
            - vocab_entries: List of vocab entries (events first, then parents)
            - num_event_codes: Number of event codes (= num_ntp_classes for joint)
        """
        vocab = []

        for token_string, weight in self.combined_weights.items():
            # Bound weight for entropy calculation
            if weight >= 1.0:
                weight = 1.0 - 1e-8
            if weight <= 0.0:
                weight = 1e-8

            entry = {
                "type": "code",
                "code_string": token_string,
                "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
            }
            vocab.append(entry)

        # Sort by weight (ascending = most informative first)
        vocab.sort(key=lambda x: x["weight"])

        # Truncate event codes to vocab_size
        vocab = vocab[: self.vocab_size]
        num_event_codes = len(vocab)

        return vocab, num_event_codes

    def save(
        self,
        output_path: PathLike,
        save_reservoirs: bool = False,
        overwrite: bool = False,
        precomputed_vocab: tuple[list[dict[str, Any]], int] | None = None,
    ) -> None:
        """Save joint vocabulary with metadata."""
        output_path = PathValidator(path=output_path).path
        output_path.mkdir(exist_ok=True, parents=True)

        if precomputed_vocab is not None:
            vocab_entries, num_event_codes = precomputed_vocab
        else:
            vocab_entries, num_event_codes = self.get_vocab()

        result = {
            "vocab": vocab_entries,
            "age_stats": {
                "mean": self.age_stats.mean,
                "std": self.age_stats.std,
            },
            "config": {
                "tokenization_mode": "joint",
                "num_ntp_classes": num_event_codes,
                "num_event_codes": num_event_codes,
                "num_parent_codes": 0,
                "num_quantiles": self.num_quantiles,
                "emit_quantiles": self.emit_quantiles,
                "emit_text": self.emit_text,
                "emit_stage": self.emit_stage,
                "remove_prefixes": self.remove_prefixes,
                "separator": self.separator,
            },
            "quantile_breaks": self.quantile_breaks,
            "discovered_stages": sorted(self.discovered_stages),
        }

        write_dict_to_json(data=result, path=output_path / "vocab.json", overwrite=overwrite)

        if save_reservoirs:
            reservoirs = {
                "combined_weights": dict(self.combined_weights),
                "discovered_stages": sorted(self.discovered_stages),
                "base_codes": [],
                "parent_codes": [],
            }
            write_dict_to_json(data=reservoirs, path=output_path / "reservoirs.json", overwrite=overwrite)
