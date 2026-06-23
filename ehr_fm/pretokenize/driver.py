"""Public pretokenize entry point.

Orchestrates dataset reading, the worker pool, sequence assembly, and Parquet
writing. The worker module is imported both for its task functions and to reset
its process-global state at the end of a run.
"""

import os
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm.auto import tqdm

from ehr_fm.defaults import DEFAULT_INTERVAL_LABELS, DEFAULT_SEX_CODES
from ehr_fm.io import read_json_yaml
from ehr_fm.logger import setup_logging
from ehr_fm.pretokenize import worker
from ehr_fm.pretokenize.worker import _pretokenize_init_worker, _pretokenize_process_row
from ehr_fm.pretokenize.writer import _pretokenize_flush
from ehr_fm.types import PathLike


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
    quantile_breaks_override_path: PathLike | None = None,
):
    """Pretokenize MEDS data and write the result as Parquet."""

    from ehr_fm.data import create_dataset

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
            quantile_breaks_override_path,
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
            quantile_breaks_override_path,
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

    # Reset the worker module's process-global state (set in-process by the
    # sequential path) so a subsequent call starts clean.
    worker._worker_state = None

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
