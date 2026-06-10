"""Public entry point for *embedding*-mode pretokenization.

'Embedding' here is the model input mode (events -> embedding_text_id + NTP label
+ numeric feature vector), NOT ehr_fm.embedding, which builds the text-embedding
table. Mirrors ehr_fm.pretokenize.driver: orchestrates dataset reading, the worker
pool, and Parquet writing. The thin scripts/pretokenize_embedding.py CLI delegates
its whole body here.
"""

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm.auto import tqdm

from ehr_fm.data import create_dataset
from ehr_fm.logger import setup_logging
from ehr_fm.pretokenize.embedding_worker import _init_worker, _process_row
from ehr_fm.pretokenize.embedding_writer import _embedding_flush
from ehr_fm.types import PathLike


def pretokenize_embedding_data(
    vocab_path: PathLike,
    embedding_lookup_path: PathLike,
    out_dir: PathLike,
    dataset_path: PathLike,
    samples_path: PathLike | None = None,
    split: str | None = None,
    numeric_stats_path: PathLike | None = None,
    numeric_pathway_mode: str = "legacy_zscore",
    numeric_override_mode: str = "none",
    numeric_quantile_breaks_path: PathLike | None = None,
    num_workers: int = -1,
    vocab_size: int | None = None,
    row_group_size: int = 32768,
    output_filename: str = "patients_tokenized.parquet",
):
    """Pretokenize MEDS data for the embedding input mode and write Parquet."""
    logger = setup_logging(child_name="pretokenize_embedding")

    dataset_path = Path(dataset_path)

    # Validate embedding_text exists in MEDS Reader
    emb_text_dir = dataset_path / "embedding_text"
    if not emb_text_dir.is_dir():
        logger.warning(
            f"embedding_text directory not found at {emb_text_dir}. "
            "Ensure MEDS data was generated with generate_embedding_text: true."
        )

    samples_path = samples_path or str(dataset_path / "metadata" / "samples.parquet")
    dataset_config = {
        "dataset_path": str(dataset_path),
        "samples_path": samples_path,
        "split": split,
    }
    dataset = create_dataset(dataset_config)
    logger.info(f"Dataset: {len(dataset)} samples")

    os.makedirs(out_dir, exist_ok=True)
    effective_num_workers = cpu_count() if num_workers == -1 else num_workers

    process_fn = partial(_process_row, vocab_size=vocab_size)

    init_args = (
        vocab_path,
        embedding_lookup_path,
        numeric_stats_path,
        vocab_size,
        numeric_override_mode,
        numeric_quantile_breaks_path,
        numeric_pathway_mode,
    )

    if effective_num_workers > 1:
        pool = Pool(effective_num_workers, initializer=_init_worker, initargs=init_args)
        results = pool.imap_unordered(process_fn, dataset)
    else:
        _init_worker(*init_args)
        results = map(process_fn, dataset)
        pool = None

    writer = None
    batch = []
    written = 0

    with tqdm(total=len(dataset), desc="pretokenize_embedding") as bar:
        for out in results:
            bar.update()
            if out is None:
                continue
            batch.append(out)
            written += 1
            if len(batch) >= row_group_size:
                writer = _embedding_flush(batch, writer, out_dir, file_name=output_filename)
                batch = []

    if batch:
        _embedding_flush(batch, writer, out_dir, file_name=output_filename, final=True)
    elif writer:
        writer.close()

    if pool:
        pool.close()
        pool.join()

    logger.info(f"Wrote {written} samples to {out_dir}/{output_filename}")
