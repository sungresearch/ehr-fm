"""Pool worker for pretokenization.

All per-process state is bundled into a single ``_WorkerState`` object, built
once by ``_pretokenize_init_worker`` -- run as the multiprocessing ``Pool``
initializer in each child process -- and read by ``_pretokenize_process_row``.
The initializer and the per-row reader deliberately live in the SAME module so
they share one module-global (``_worker_state``) per worker process; splitting
them apart would break that contract.
"""

import dataclasses
import datetime

import pyarrow as pa

from ehr_fm.defaults import (
    DEFAULT_INTERVAL_BINS,
    DEFAULT_INTERVAL_LABELS,
    DEFAULT_SEX_CODES,
    DEMO_YEAR_MAX,
    DEMO_YEAR_MIN,
    REPEATABLE_INTERVAL_LABEL,
)
from ehr_fm.io import read_json_yaml
from ehr_fm.logger import setup_logging
from ehr_fm.pretokenize.config import PretokenizeWorkerConfig
from ehr_fm.pretokenize.lookups import (
    _build_interval_token_lookup,
    _build_token_string_lookup,
    _tokenize_event_with_policy,
)
from ehr_fm.tokenization import (
    FactorizedConfig,
    FactorizedPolicy,
    JointConfig,
    JointPolicy,
    TokenizationPolicy,
)
from ehr_fm.types import PathLike


@dataclasses.dataclass(frozen=True)
class _WorkerState:
    """Immutable per-process pretokenize state, built once by the initializer."""

    policy: TokenizationPolicy
    token_lookup: dict[str, int]
    config: PretokenizeWorkerConfig
    demo_lookup: dict[str, int]


# Set by _pretokenize_init_worker (pool initializer, or the sequential path in
# the driver), read by _pretokenize_process_row, reset by the driver. ``None``
# until initialized -- defining it at module scope keeps the guard below a plain
# ``is None`` check rather than relying on the name having been created.
_worker_state: "_WorkerState | None" = None


def _pretokenize_init_worker(
    vocab_path: PathLike,
    inject_time_intervals: bool = False,
    max_interval_repeat: int | None = None,
    demographic_prefix: bool = False,
    demographic_include_year: bool = False,
    sex_codes_male: str | None = None,
    sex_codes_female: str | None = None,
    sex_codes_unknown: str | None = None,
    quantile_breaks_override_path: PathLike | None = None,
):
    global _worker_state

    logger = setup_logging(child_name="pretokenize_worker")
    vocab_data = read_json_yaml(vocab_path)
    vocab_entries = vocab_data["vocab"]
    age_stats = vocab_data["age_stats"]

    interval_token_lookup = _build_interval_token_lookup(vocab_entries)

    worker_config = PretokenizeWorkerConfig(
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

    demo_lookup = {}
    for i, v in enumerate(vocab_entries):
        if v.get("type") == "demographic":
            label = v.get("label")
            if label:
                demo_lookup[label] = i

    vocab_config = vocab_data.get("config", {})
    tokenization_mode = vocab_config.get("tokenization_mode")

    quantile_breaks = vocab_data.get("quantile_breaks", {})

    if quantile_breaks_override_path:
        override_data = read_json_yaml(quantile_breaks_override_path)
        if isinstance(override_data, dict) and "quantile_breaks" in override_data:
            override_breaks = override_data["quantile_breaks"]
        else:
            override_breaks = override_data
        replaced = 0
        for code, breaks in override_breaks.items():
            if code in quantile_breaks:
                quantile_breaks[code] = breaks
                replaced += 1
        logger.info(
            f"Quantile breaks override: replaced {replaced}/{len(override_breaks)} codes "
            f"(overlap-only with {len(quantile_breaks)} vocab codes)"
        )

    known_stages = set(vocab_data.get("discovered_stages", []))

    token_lookup = _build_token_string_lookup(vocab_entries)

    if tokenization_mode == "factorized":
        factorized_config = FactorizedConfig(
            emit_quantiles=vocab_config.get("emit_quantiles", True),
            emit_text=vocab_config.get("emit_text", True),
            emit_stage=vocab_config.get("emit_stage", True),
            num_quantiles=vocab_config.get("num_quantiles", 10),
            remove_prefixes=vocab_config.get("remove_prefixes", True),
            separator=vocab_config.get("separator", "/"),
        )
        policy_instance = FactorizedPolicy(
            config=factorized_config,
            quantile_breaks=quantile_breaks,
            known_stages=known_stages,
            token_lookup=token_lookup,
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
        policy_instance = JointPolicy(
            config=joint_config,
            quantile_breaks=quantile_breaks,
            known_stages=known_stages,
            token_lookup=token_lookup,
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

    _worker_state = _WorkerState(
        policy=policy_instance,
        token_lookup=token_lookup,
        config=worker_config,
        demo_lookup=demo_lookup,
    )


def _pretokenize_process_row(row, *, current_max_events: int, vocab_size: int):
    if row is None or _worker_state is None:
        return None

    policy = _worker_state.policy
    token_lookup = _worker_state.token_lookup
    config = _worker_state.config
    demo_lookup = _worker_state.demo_lookup

    pid, seq = row

    try:
        birth_t = next(e["time"] for e in seq if e["code"] == "MEDS_BIRTH")
    except StopIteration:
        if not seq:
            return {"_skip_reason": "empty_sequence"}
        birth_t = seq[0]["time"]

    vocab_size = vocab_size or config.vocab_size

    codes, ages, ages_normalized = [], [], []
    prev_time = None
    inject_intervals = config.inject_time_intervals
    bins = config.interval_bins
    max_repeat = config.max_interval_repeat
    interval_lookup = config.interval_token_lookup
    demographic_prefix = config.demographic_prefix

    if current_max_events is not None and demographic_prefix:
        raise ValueError(
            "max_events_per_patient and demographic_prefix cannot be used together. "
            "max_events_per_patient keeps the LAST (most recent) events, which excludes "
            "MEDS_BIRTH and sex codes needed for demographic prefix tokens. "
            "Disable one of these options."
        )

    events_to_process = seq if current_max_events is None else seq[-current_max_events:]
    male_override = config.sex_codes_male
    female_override = config.sex_codes_female
    unknown_override = config.sex_codes_unknown
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

        demographic_include_year = config.demographic_include_year
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
            tok = demo_lookup.get(lbl)
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
        if policy is not None and token_lookup is not None:
            event_tokens = _tokenize_event_with_policy(m, policy, token_lookup, vocab_size)

            if not event_tokens:
                continue

            event_time = m["time"]
            time_diff = event_time - birth_t
            dt_val = time_diff / datetime.timedelta(days=1)
            dt_normalized = config.normalize_age(time_diff)

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
                                        config.normalize_age(datetime.timedelta(seconds=inserted_age_seconds))
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
                            ages_normalized.append(config.normalize_age(event_time - birth_t))

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
