"""
ehr_fm.tokenizer
==================

Overview
--------
Pretokenization pipeline: reads a vocabulary JSON, selects the appropriate
tokenization policy (imported from ``ehr_fm.tokenization``), and converts MEDS
events into training‑ready Parquet files.

Components (defined in this module)
------------------------------------
- ``PretokenizeWorkerConfig``: Frozen dataclass holding per‑worker configuration
  (vocab_size, age normalization stats, interval/demographic settings).

- ``_tokenize_event_with_policy()``: Bridges a single MEDS event to the imported
  ``TokenizationPolicy``, enforcing the no‑orphans invariant (if the base token is
  OOV, all attribute tokens for that event are dropped) and vocab‑cutoff filtering.

- ``_pretokenize_process_row()``: Per-patient sequence assembly: events, intervals,
  demographics, ages

- ``pretokenize_data(...)``: Public entry point.  Orchestrates dataset reading,
  parallel workers, sequence assembly, optional interval injection, demographic
  prefix, vocab cutoff, and Parquet writing.

Tokenization policies (defined in ``ehr_fm.tokenization``, used here)
----------------------------------------------------------------------
- ``JointPolicy``: emits a single combined token per event (e.g., "glucose/Q:3").
- ``FactorizedPolicy``: emits separate base + attribute tokens (e.g., ["glucose", "Q:3"]).
- The appropriate policy is instantiated in ``_pretokenize_init_worker`` based on the
  ``tokenization_mode`` field in the vocabulary config.

Temporal interval tokens
------------------------
- Enabled by CLI flags passed through pretokenize:
  --inject_time_intervals, --max_interval_repeat.
- Default scheme bins (min‑inclusive, max‑exclusive), month = 30 days:
  5m–15m, 15m–1h, 1h–2h, 2h–6h, 6h–12h, 12h–1d,
  1d–3d, 3d–1w, 1w–2w, 2w–1mt, 1mt–3mt, 3mt–6mt, and repeatable 6mt.
  (12 ranged bins + 1 repeatable label = 13 interval tokens total.)
- Policy:
  - No interval before the first non‑birth event.
  - If Δt < smallest bin (5m), insert none.
  - For Δt ≥ 6 months, emit ceil(Δt / 6mt) repeats of INT_6mt (capped by
    --max_interval_repeat if set).
  - For 5m ≤ Δt < 6 months, emit the single matching bin token.
- Interval tokens are inserted into the token stream with their own age entries.
"""
import dataclasses
import datetime
import os
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .defaults import (
    DEFAULT_INTERVAL_BINS,
    DEFAULT_INTERVAL_LABELS,
    DEFAULT_SEX_CODES,
    DEMO_YEAR_MAX,
    DEMO_YEAR_MIN,
    REPEATABLE_INTERVAL_LABEL,
)
from .io import read_json_yaml
from .logger import setup_logging
from .tokenization import (
    FactorizedConfig,
    FactorizedPolicy,
    JointConfig,
    JointPolicy,
    TokenizationPolicy,
)
from .types import PathLike


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


def _build_token_string_lookup(vocab: list[dict[str, Any]]) -> dict[str, int]:
    """Build a lookup from token string to vocab index for factorized tokenization.

    Maps:
    - code entries: code_string → index
    - quantile entries: label (Q:*) → index
    - stage entries: label (STAGE:*) → index
    - text entries: label (TXT:*) → index
    - interval entries: label → index
    - demographic entries: label → index
    """
    lookup: dict[str, int] = {}
    for i, entry in enumerate(vocab):
        entry_type = entry.get("type")
        if entry_type == "code":
            code_string = entry.get("code_string")
            if code_string:
                lookup[code_string] = i
        elif entry_type in ("quantile", "stage", "interval", "demographic"):
            label = entry.get("label")
            if label:
                lookup[label] = i
        elif entry_type == "text":
            # Text entries use TXT:<normalized> format
            text_string = entry.get("text_string")
            if text_string:
                lookup[f"TXT:{text_string}"] = i
        # numeric entries are not used in factorized mode (use Q:* instead)
    return lookup


def _pretokenize_init_worker(
    vocab_path: PathLike,
    inject_time_intervals: bool = False,
    max_interval_repeat: int | None = None,
    demographic_prefix: bool = False,
    demographic_include_year: bool = False,
    sex_codes_male: str | None = None,
    sex_codes_female: str | None = None,
    sex_codes_unknown: str | None = None,
):
    global _pretok_policy_instance, _pretok_token_lookup
    global _pretok_worker_config, _pretok_demo_lookup

    logger = setup_logging(child_name="pretokenize_worker")
    vocab_data = read_json_yaml(vocab_path)
    vocab_entries = vocab_data["vocab"]
    age_stats = vocab_data["age_stats"]

    interval_token_lookup = _build_interval_token_lookup(vocab_entries)

    _pretok_worker_config = PretokenizeWorkerConfig(
        vocab_size=len(vocab_entries),
        age_mean=age_stats["mean"],
        age_std=age_stats["std"],
        inject_time_intervals=inject_time_intervals,
        interval_bins=DEFAULT_INTERVAL_BINS,
        max_interval_repeat=max_interval_repeat,
        interval_token_lookup=interval_token_lookup,
        demographic_prefix=demographic_prefix,
        demographic_include_year=demographic_include_year,
        sex_codes_male=sex_codes_male,
        sex_codes_female=sex_codes_female,
        sex_codes_unknown=sex_codes_unknown,
    )

    _pretok_demo_lookup = {}
    for i, v in enumerate(vocab_entries):
        if v.get("type") == "demographic":
            label = v.get("label")
            if label:
                _pretok_demo_lookup[label] = i

    vocab_config = vocab_data.get("config", {})
    tokenization_mode = vocab_config.get("tokenization_mode")

    quantile_breaks = vocab_data.get("quantile_breaks", {})
    known_stages = set(vocab_data.get("discovered_stages", []))

    _pretok_token_lookup = _build_token_string_lookup(vocab_entries)

    if tokenization_mode == "factorized":
        factorized_config = FactorizedConfig(
            emit_quantiles=vocab_config.get("emit_quantiles", True),
            emit_text=vocab_config.get("emit_text", True),
            emit_stage=vocab_config.get("emit_stage", True),
            num_quantiles=vocab_config.get("num_quantiles", 10),
            remove_prefixes=vocab_config.get("remove_prefixes", True),
            separator=vocab_config.get("separator", "/"),
        )
        _pretok_policy_instance = FactorizedPolicy(
            config=factorized_config,
            quantile_breaks=quantile_breaks,
            known_stages=known_stages,
            token_lookup=_pretok_token_lookup,
        )
    else:
        joint_config = JointConfig(
            emit_quantiles=vocab_config.get("emit_quantiles", True),
            emit_text=vocab_config.get("emit_text", True),
            emit_stage=vocab_config.get("emit_stage", True),
            num_quantiles=vocab_config.get("num_quantiles", 10),
            remove_prefixes=vocab_config.get("remove_prefixes", True),
            separator=vocab_config.get("separator", "/"),
        )
        _pretok_policy_instance = JointPolicy(
            config=joint_config,
            quantile_breaks=quantile_breaks,
            known_stages=known_stages,
            token_lookup=_pretok_token_lookup,
        )

    if inject_time_intervals:
        expected_labels = set(DEFAULT_INTERVAL_LABELS)
        present_labels = {v.get("label") for v in vocab_entries if v.get("type") == "interval"}
        missing = sorted(expected_labels - present_labels)
        if missing:
            raise ValueError(
                "inject_time_intervals=True but required interval tokens are missing in vocab: "
                + ", ".join(missing)
                + ". Retrain vocabulary with --add_time_interval_tokens."
            )

    if demographic_prefix:
        expected_demo = set(
            [
                "DEMO_SEX_MALE",
                "DEMO_SEX_FEMALE",
                "DEMO_SEX_UNKNOWN",
            ]
            + [f"DEMO_AGE_{start}_{start+4}" for start in range(0, 100, 5)]
            + ["DEMO_AGE_100P"]
        )
        if demographic_include_year:
            expected_demo.update([f"DEMO_YEAR_{y}_{y+4}" for y in range(DEMO_YEAR_MIN, DEMO_YEAR_MAX + 1, 5)])
        present_demo = {v.get("label") for v in vocab_entries if v.get("type") == "demographic"}
        missing = sorted(expected_demo - present_demo)
        if missing:
            raise ValueError(
                "demographic_prefix=True but required demographic tokens are missing in vocab: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
                + ". Retrain vocabulary with --add_demographic_tokens."
            )

    num_interval_present = sum(1 for v in vocab_entries if v.get("type") == "interval")
    num_demo_present = sum(1 for v in vocab_entries if v.get("type") == "demographic")
    logger.info(
        "Pretokenize configuration: "
        f"inject_time_intervals={inject_time_intervals}, demographic_prefix={demographic_prefix}, "
        f"demographic_include_year={demographic_include_year}, "
        f"max_interval_repeat={max_interval_repeat}, "
        f"interval_tokens_present={num_interval_present}/{len(DEFAULT_INTERVAL_LABELS)}, "
        f"demographic_tokens_present={num_demo_present}"
    )


def _build_interval_token_lookup(vocab: list[dict[str, Any]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, v in enumerate(vocab):
        if v.get("type") == "interval":
            lbl = v.get("label")
            if isinstance(lbl, str):
                lookup[lbl] = i
    return lookup


def _tokenize_event_with_policy(
    event: dict[str, Any],
    policy: TokenizationPolicy,
    token_lookup: dict[str, int],
    vocab_size: int,
) -> list[int]:
    """Tokenize a single event using a tokenization policy.

    Works for both joint and factorized modes:
    - JointPolicy: emits single combined token like "4096101/Q:7"
    - FactorizedPolicy: emits multiple tokens like ["4096101", "Q:7"]

    Returns list of token IDs. Empty list if base token is OOV (no-orphans invariant).

    Args:
        event: Event dict with code, numeric_value, text_value, workflow_stage
        policy: TokenizationPolicy instance (JointPolicy or FactorizedPolicy)
        token_lookup: token_string → vocab_index mapping
        vocab_size: Maximum token ID (for cutoff filtering)

    Returns:
        List of valid token IDs for this event
    """
    token_strings = policy.emit_token_strings(event)
    if not token_strings:
        return []

    # First token is always the base token - check for OOV
    base_token_str = token_strings[0]
    base_tok_id = token_lookup.get(base_token_str)

    # No-orphans invariant: if base token is OOV, skip all tokens for this event
    if base_tok_id is None or base_tok_id >= vocab_size:
        return []

    # Collect valid token IDs
    valid_tokens = [base_tok_id]
    for token_str in token_strings[1:]:
        tok_id = token_lookup.get(token_str)
        if tok_id is not None and tok_id < vocab_size:
            valid_tokens.append(tok_id)

    return valid_tokens


def _pretokenize_process_row(row, *, current_max_events: int, vocab_size: int):
    global _pretok_policy_instance, _pretok_token_lookup
    global _pretok_worker_config, _pretok_demo_lookup
    if row is None or _pretok_worker_config is None:
        return None

    pid, seq = row

    try:
        birth_t = next(e["time"] for e in seq if e["code"] == "MEDS_BIRTH")
    except StopIteration:
        if not seq:
            return {"_skip_reason": "empty_sequence"}
        birth_t = seq[0]["time"]

    vocab_size = vocab_size or _pretok_worker_config.vocab_size

    codes, ages, ages_normalized = [], [], []
    prev_time = None
    inject_intervals = _pretok_worker_config.inject_time_intervals
    bins = _pretok_worker_config.interval_bins
    max_repeat = _pretok_worker_config.max_interval_repeat
    interval_lookup = _pretok_worker_config.interval_token_lookup
    demographic_prefix = _pretok_worker_config.demographic_prefix

    if current_max_events is not None and demographic_prefix:
        raise ValueError(
            "max_events_per_patient and demographic_prefix cannot be used together. "
            "max_events_per_patient keeps the LAST (most recent) events, which excludes "
            "MEDS_BIRTH and sex codes needed for demographic prefix tokens. "
            "Disable one of these options."
        )

    events_to_process = seq if current_max_events is None else seq[-current_max_events:]
    male_override = _pretok_worker_config.sex_codes_male
    female_override = _pretok_worker_config.sex_codes_female
    unknown_override = _pretok_worker_config.sex_codes_unknown
    sex_codes_male = set(male_override.split(",")) if male_override else set(DEFAULT_SEX_CODES["male"])
    sex_codes_female = (
        set(female_override.split(",")) if female_override else set(DEFAULT_SEX_CODES["female"])
    )
    sex_codes_unknown = (
        set(unknown_override.split(",")) if unknown_override else set(DEFAULT_SEX_CODES["unknown"])
    )

    # helpers for intervals
    minute = 60.0
    hour = 60.0 * minute
    day = 24.0 * hour
    month = 30.0 * day

    def _repeatable_6mt_tokens(delta_seconds: float) -> int:
        if delta_seconds <= 0:
            return 0
        count = int((delta_seconds + (6 * month) - 1) // (6 * month))  # ceil
        if max_repeat is not None:
            count = min(count, int(max_repeat))
        return count

    def _find_bin_token(delta_seconds: float) -> int | None:
        for lo, hi, label in bins:
            if lo <= delta_seconds < hi:
                return interval_lookup.get(label)
        return None

    INT_6MT_ID = interval_lookup.get(REPEATABLE_INTERVAL_LABEL)

    # Demographic prefix logic
    first_event_idx = 0
    if demographic_prefix:
        # Identify sex and first non-demographic event
        sex_label = "DEMO_SEX_UNKNOWN"
        try:
            birth_t = next(e["time"] for e in events_to_process if e["code"] == "MEDS_BIRTH")
        except StopIteration:
            birth_t = events_to_process[0]["time"] if events_to_process else None

        first_event_idx = None
        for i, e in enumerate(events_to_process):
            code = e.get("code")
            if code == "MEDS_BIRTH":
                continue
            if code in sex_codes_male:
                sex_label = "DEMO_SEX_MALE"
                continue
            if code in sex_codes_female:
                sex_label = "DEMO_SEX_FEMALE"
                continue
            if code in sex_codes_unknown and sex_label == "DEMO_SEX_UNKNOWN":
                sex_label = "DEMO_SEX_UNKNOWN"
                continue
            first_event_idx = i
            break
        if first_event_idx is None:
            # no medical events after birth/sex → drop sample
            return {"_skip_reason": "no_medical_events_after_birth_or_sex"}

        first_event = events_to_process[first_event_idx]
        age_years = int(((first_event["time"] - birth_t).total_seconds()) // (365.25 * 24 * 3600))
        if age_years < 0:
            age_years = 0
        if age_years >= 100:
            age_label = "DEMO_AGE_100P"
        else:
            bucket_start = (age_years // 5) * 5
            bucket_end = bucket_start + 4
            age_label = f"DEMO_AGE_{bucket_start}_{bucket_end}"

        demo_labels = [sex_label, age_label]

        demographic_include_year = _pretok_worker_config.demographic_include_year
        if demographic_include_year:
            # Year bucket at first event
            year = first_event["time"].year
            if year < DEMO_YEAR_MIN or year > DEMO_YEAR_MAX:
                _logger = setup_logging(child_name="pretokenize_worker")
                _logger.warning(
                    f"First event year {year} outside [{DEMO_YEAR_MIN}, {DEMO_YEAR_MAX}]; check data. "
                    f"Event code: {first_event['code']}, Event time: {first_event['time']}, "
                    f"Event description: {first_event['description']}"
                )
                return {"_skip_reason": "first_event_year_out_of_range"}
            year_bucket_start = ((year - DEMO_YEAR_MIN) // 5) * 5 + DEMO_YEAR_MIN
            year_bucket_end = year_bucket_start + 4
            year_label = f"DEMO_YEAR_{year_bucket_start}_{year_bucket_end}"
            demo_labels.append(year_label)

        for lbl in demo_labels:
            tok = _pretok_demo_lookup.get(lbl)
            if tok is None or tok >= vocab_size:
                _logger = setup_logging(child_name="pretokenize_worker")
                _logger.warning(f"Demographic token {lbl} missing or >= vocab_size; skipping sample.")
                return {"_skip_reason": "missing_demographic_token"}
            codes.append(tok)

        # Ages as positions; age_normalized zeros for the demographic tokens
        num_demo_tokens = len(demo_labels)
        ages.extend([float(i) for i in range(num_demo_tokens)])
        ages_normalized.extend([0.0] * num_demo_tokens)
        prev_time = first_event["time"]
        start_idx = first_event_idx
    else:
        start_idx = 0

    for m in events_to_process[start_idx:]:
        if _pretok_policy_instance is not None and _pretok_token_lookup is not None:
            event_tokens = _tokenize_event_with_policy(
                m, _pretok_policy_instance, _pretok_token_lookup, vocab_size
            )

            if not event_tokens:
                continue

            event_time = m["time"]
            time_diff = event_time - birth_t
            dt_val = time_diff / datetime.timedelta(days=1)
            dt_normalized = _pretok_worker_config.normalize_age(time_diff)

            if inject_intervals and prev_time is not None:
                dt = (event_time - prev_time).total_seconds()
                smallest_min = bins[0][0] if bins else float("inf")
                if dt >= smallest_min:
                    if INT_6MT_ID is not None and INT_6MT_ID < vocab_size and dt >= 6 * month:
                        count = _repeatable_6mt_tokens(dt)
                        if count > 0:
                            prev_age_seconds = (prev_time - birth_t).total_seconds()
                            event_age_seconds = (event_time - birth_t).total_seconds()
                            for i in range(count):
                                codes.append(INT_6MT_ID)
                                if demographic_prefix:
                                    ages.append(float(len(ages)))
                                    ages_normalized.append(0.0)
                                else:
                                    inserted_age_seconds = prev_age_seconds + (i + 1) * (6 * month)
                                    if inserted_age_seconds > event_age_seconds:
                                        inserted_age_seconds = event_age_seconds
                                    ages.append(inserted_age_seconds / day)
                                    ages_normalized.append(
                                        _pretok_worker_config.normalize_age(
                                            datetime.timedelta(seconds=inserted_age_seconds)
                                        )
                                    )
                            dt = dt - count * 6 * month
                    bin_tok = _find_bin_token(dt)
                    if bin_tok is not None and bin_tok < vocab_size:
                        codes.append(bin_tok)
                        if demographic_prefix:
                            ages.append(float(len(ages)))
                            ages_normalized.append(0.0)
                        else:
                            event_age_seconds = (event_time - birth_t).total_seconds()
                            ages.append(event_age_seconds / day)
                            ages_normalized.append(_pretok_worker_config.normalize_age(event_time - birth_t))

            for tok in event_tokens:
                codes.append(tok)
                if demographic_prefix:
                    ages.append(float(len(ages)))
                    ages_normalized.append(0.0)
                else:
                    ages.append(dt_val)
                    ages_normalized.append(dt_normalized)

            prev_time = event_time
            continue

    # Periodic garbage collection in worker processes to prevent memory accumulation
    import random

    if random.randint(1, 1000) == 1:  # ~0.1% chance per task
        import gc

        gc.collect()

    arr_dict = {
        "subject_id": pid[0],
        "index_time": pid[1],
        "token_ids": pa.array(codes, type=pa.int32()),
        "age": pa.array(ages, type=pa.float32()),
        "length": len(codes),
        "age_normalized": pa.array(ages_normalized, type=pa.float32()),
    }
    return arr_dict


def _pretokenize_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
            pa.field("age_normalized", pa.list_(pa.float32())),
        ]
    )


def _pretokenize_flush(rows, writer, out_dir, file_name="patients_tokenized.parquet", final=False):
    table = pa.Table.from_pylist(rows, schema=_pretokenize_schema())
    if writer is None:
        output_path = os.path.join(out_dir, file_name)
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd", use_dictionary=True)

    writer.write_table(table)

    # Explicitly clean up table to free memory immediately
    del table

    if final:
        writer.close()
        writer = None

    return writer


def pretokenize_data(
    vocab_path: PathLike,
    out_dir: PathLike,
    dataset_path: PathLike = None,
    samples_path: PathLike | None = None,
    split: str | None = None,
    num_workers: int = -1,
    max_events_per_patient: int = None,
    row_group_size: int = 32_768,
    output_filename: str = "patients_tokenized.parquet",
    vocab_size: int = None,
    inject_time_intervals: bool = False,
    max_interval_repeat: int | None = None,
    demographic_prefix: bool = False,
    demographic_include_year: bool = False,
    sex_codes_male: str | None = None,
    sex_codes_female: str | None = None,
    sex_codes_unknown: str | None = None,
):
    """Pretokenize MEDS data and write the result as Parquet."""

    from .data import create_dataset

    logger = setup_logging(child_name="pretokenize_data")
    os.makedirs(out_dir, exist_ok=True)
    dataset_config = {"split": split}

    if not dataset_path:
        raise ValueError("Must specify dataset_path")

    # MEDS reader configuration
    dataset_config["dataset_path"] = dataset_path
    logger.info(f"Using dataset with auto-detection: {dataset_path}")

    dataset_config["samples_path"] = samples_path

    logger.info(f"Creating dataset with config: {dataset_config}")
    dataset = create_dataset(dataset_config)
    logger.info(f"Created {type(dataset).__name__} with {len(dataset)} samples")

    # Upfront configuration summary (single log from main process)
    try:
        vocab_data = read_json_yaml(vocab_path)
        vocab_entries = vocab_data.get("vocab", [])
        num_interval_present = sum(1 for v in vocab_entries if v.get("type") == "interval")
        num_demo_present = sum(1 for v in vocab_entries if v.get("type") == "demographic")
        male_codes = set(sex_codes_male.split(",")) if sex_codes_male else set(DEFAULT_SEX_CODES["male"])
        female_codes = (
            set(sex_codes_female.split(",")) if sex_codes_female else set(DEFAULT_SEX_CODES["female"])
        )
        unknown_codes = (
            set(sex_codes_unknown.split(",")) if sex_codes_unknown else set(DEFAULT_SEX_CODES["unknown"])
        )

        def _preview(s: set[str]) -> str:
            items = sorted(s)
            return ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")

        logger.info(
            "Pretokenize configuration (main): "
            f"inject_time_intervals={inject_time_intervals}, demographic_prefix={demographic_prefix}, "
            f"max_interval_repeat={max_interval_repeat}, "
            f"interval_tokens_present={num_interval_present}/{len(DEFAULT_INTERVAL_LABELS)}, "
            f"demographic_tokens_present={num_demo_present}, "
            f"sex_codes: male({len(male_codes)}: {_preview(male_codes)}), "
            f"female({len(female_codes)}: {_preview(female_codes)}), "
            f"unknown({len(unknown_codes)}: {_preview(unknown_codes)})"
        )
    except Exception:
        # Non-fatal: continue even if summary cannot be produced
        pass

    writer = None
    flush_count = 0
    processed_count = 0
    written_count = 0
    none_count = 0
    skip_reasons: dict[str, int] = {}

    effective_num_workers = cpu_count() if num_workers == -1 else num_workers

    process_row_configured = partial(
        _pretokenize_process_row,
        current_max_events=max_events_per_patient,
        vocab_size=vocab_size,
    )

    pool: Pool | None = None
    results_iterable: Iterable

    if effective_num_workers > 1:
        pool_initargs = (
            vocab_path,
            inject_time_intervals,
            max_interval_repeat,
            demographic_prefix,
            demographic_include_year,
            sex_codes_male,
            sex_codes_female,
            sex_codes_unknown,
        )
        pool = Pool(effective_num_workers, initializer=_pretokenize_init_worker, initargs=pool_initargs)
        results_iterable = pool.imap_unordered(process_row_configured, dataset)
    else:
        _pretokenize_init_worker(
            vocab_path,
            inject_time_intervals,
            max_interval_repeat,
            demographic_prefix,
            demographic_include_year,
            sex_codes_male,
            sex_codes_female,
            sex_codes_unknown,
        )
        results_iterable = map(process_row_configured, dataset)

    flat_batch = []

    with tqdm(total=len(dataset), desc="pretokenize_data") as bar:
        for out in results_iterable:
            bar.update()
            processed_count += 1
            if out is None:
                none_count += 1
                continue
            if isinstance(out, dict) and "_skip_reason" in out:
                reason = out["_skip_reason"]
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            flat_batch.append(out)
            written_count += 1

            if len(flat_batch) == row_group_size:
                writer = _pretokenize_flush(flat_batch, writer, out_dir, file_name=output_filename)
                flat_batch = []
                flush_count += 1

    # Final flush
    if flat_batch:
        _pretokenize_flush(flat_batch, writer, out_dir, file_name=output_filename, final=True)
    elif writer:
        writer.close()

    if pool:
        pool.close()
        pool.join()

    global _pretok_worker_config, _pretok_demo_lookup
    _pretok_worker_config = None
    _pretok_demo_lookup = None

    # End-of-run summary
    logger.info("Pretokenization Statistics:")
    logger.info(f"  Total samples processed: {processed_count}")
    logger.info(f"  Samples with valid data: {written_count}")
    total_skipped = processed_count - written_count
    logger.info(f"  Samples skipped (total): {total_skipped}")
    if none_count:
        logger.info(f"    - None outputs (other): {none_count}")
    for k, v in sorted(skip_reasons.items()):
        logger.info(f"    - {k}: {v}")
