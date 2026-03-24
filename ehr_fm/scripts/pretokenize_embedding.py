"""Embedding-mode pretokenization: produces event-level (1 per event) parquet.

Each event maps to:
- embedding_text_id: index into EmbeddingLookup table
- token_id: NTP label (joint-style vocab ID, 1 per event)
- numeric_features: 5-dim feature vector
- age / age_normalized: days since birth / z-scored
"""

import argparse
import datetime
import json
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ehr_fm.data import create_dataset
from ehr_fm.io import read_json_yaml
from ehr_fm.logger import setup_logging
from ehr_fm.tokenization import JointConfig, JointPolicy

# Module-level worker state
_worker_state = None


def _init_worker(
    vocab_path,
    embedding_lookup_path,
    numeric_stats_path,
    vocab_size,
):
    global _worker_state

    vocab_data = read_json_yaml(vocab_path)
    vocab_entries = vocab_data["vocab"]
    age_stats = vocab_data["age_stats"]
    vocab_config = vocab_data.get("config", {})

    quantile_breaks = vocab_data.get("quantile_breaks", {})
    known_stages = set(vocab_data.get("discovered_stages", []))

    # Build token string → ID lookup
    token_lookup = {}
    for i, entry in enumerate(vocab_entries):
        entry_type = entry.get("type")
        if entry_type == "code":
            code_string = entry.get("code_string")
            if code_string:
                token_lookup[code_string] = i
        elif entry_type in ("quantile", "stage", "interval", "demographic"):
            label = entry.get("label")
            if label:
                token_lookup[label] = i
        elif entry_type == "text":
            text_string = entry.get("text_string")
            if text_string:
                token_lookup[f"TXT:{text_string}"] = i

    joint_config = JointConfig(
        emit_quantiles=vocab_config.get("emit_quantiles", True),
        emit_text=vocab_config.get("emit_text", True),
        emit_stage=vocab_config.get("emit_stage", True),
        num_quantiles=vocab_config.get("num_quantiles", 10),
        remove_prefixes=vocab_config.get("remove_prefixes", True),
        separator=vocab_config.get("separator", "/"),
    )
    policy = JointPolicy(
        config=joint_config,
        quantile_breaks=quantile_breaks,
        known_stages=known_stages,
        token_lookup=token_lookup,
    )

    with open(Path(embedding_lookup_path) / "id_mapping.json") as f:
        embedding_text_to_id = json.load(f)

    numeric_stats = {}
    if numeric_stats_path and os.path.exists(numeric_stats_path):
        with open(numeric_stats_path) as f:
            numeric_stats = json.load(f)

    effective_vocab_size = vocab_size if vocab_size else len(vocab_entries)

    _worker_state = {
        "policy": policy,
        "token_lookup": token_lookup,
        "embedding_text_to_id": embedding_text_to_id,
        "numeric_stats": numeric_stats,
        "quantile_breaks": quantile_breaks,
        "age_mean": age_stats["mean"],
        "age_std": age_stats["std"],
        "vocab_size": effective_vocab_size,
    }


def _compute_numeric_features(event, code, numeric_stats, quantile_breaks):
    """Compute 5-dim numeric feature vector for an event."""
    numeric_value = event.get("numeric_value")
    if numeric_value is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # value_present
    value_present = 1.0

    # value_log_zscore
    log_val = math.log1p(abs(numeric_value))
    stats = numeric_stats.get(code)
    if stats:
        log_mean = stats.get("log_mean", 0.0)
        log_std = stats.get("log_std", 1.0)
        value_log_zscore = (log_val - log_mean) / log_std
    else:
        value_log_zscore = log_val

    # quantile
    breaks = quantile_breaks.get(code, [])
    if breaks:
        q = 0
        for b in breaks:
            if numeric_value <= b:
                break
            q += 1
        # Normalize to [0, 1]
        quantile = q / max(len(breaks), 1)
    else:
        quantile = 0.5

    # ref_range_position and ref_range_available (per-event from MEDS data)
    ref_low = event.get("ref_low")
    ref_high = event.get("ref_high")

    if ref_low is not None and ref_high is not None and ref_high > ref_low:
        ref_range_available = 1.0
        ref_range_position = (numeric_value - ref_low) / (ref_high - ref_low)
        # Clamp to [-2, 3] to handle extreme outliers
        ref_range_position = max(-2.0, min(3.0, ref_range_position))
    else:
        ref_range_available = 0.0
        ref_range_position = 0.0

    return [value_log_zscore, quantile, ref_range_position, ref_range_available, value_present]


def _process_row(row, *, vocab_size):
    global _worker_state
    if row is None or _worker_state is None:
        return None

    pid, seq = row

    # Find birth time
    try:
        birth_t = next(e["time"] for e in seq if e["code"] == "MEDS_BIRTH")
    except StopIteration:
        if not seq:
            return None
        birth_t = seq[0]["time"]

    policy = _worker_state["policy"]
    token_lookup = _worker_state["token_lookup"]
    embedding_text_to_id = _worker_state["embedding_text_to_id"]
    numeric_stats = _worker_state["numeric_stats"]
    quantile_breaks = _worker_state["quantile_breaks"]
    age_mean = _worker_state["age_mean"]
    age_std = _worker_state["age_std"]
    effective_vocab_size = vocab_size or _worker_state["vocab_size"]

    emb_ids = []
    token_ids = []
    numeric_features = []
    ages = []
    ages_normalized = []

    for event in seq:
        code = event.get("code", "")

        # Get embedding_text → embedding_text_id
        embedding_text = event.get("embedding_text")
        if embedding_text is None:
            continue
        emb_id = embedding_text_to_id.get(embedding_text)
        if emb_id is None:
            continue

        # Get NTP label via JointPolicy
        token_strings = policy.emit_token_strings(event)
        if not token_strings:
            tok_id = -100  # OOV
        else:
            tok_id = token_lookup.get(token_strings[0])
            if tok_id is None or tok_id >= effective_vocab_size:
                tok_id = -100

        # Compute numeric features
        num_feat = _compute_numeric_features(event, code, numeric_stats, quantile_breaks)

        # Compute age
        time_diff = event["time"] - birth_t
        age_days = time_diff / datetime.timedelta(days=1)
        age_norm = (time_diff.total_seconds() - age_mean) / age_std

        emb_ids.append(emb_id)
        token_ids.append(tok_id)
        numeric_features.append(num_feat)
        ages.append(age_days)
        ages_normalized.append(age_norm)

    if not emb_ids:
        return None

    return {
        "subject_id": pid[0],
        "index_time": pid[1],
        "embedding_text_ids": pa.array(emb_ids, type=pa.int32()),
        "token_ids": pa.array(token_ids, type=pa.int32()),
        "numeric_features": [pa.array(nf, type=pa.float32()) for nf in numeric_features],
        "age": pa.array(ages, type=pa.float32()),
        "age_normalized": pa.array(ages_normalized, type=pa.float32()),
        "length": len(emb_ids),
    }


def _schema():
    return pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("embedding_text_ids", pa.list_(pa.int32())),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("numeric_features", pa.list_(pa.list_(pa.float32()))),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("age_normalized", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Embedding-mode pretokenization.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MEDS Reader dataset.")
    parser.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to samples.parquet.",
    )
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json.")
    parser.add_argument(
        "--embedding_lookup_path", type=str, required=True, help="Path to embedding lookup artifacts dir."
    )
    parser.add_argument(
        "--numeric_stats_path", type=str, default=None, help="Path to numeric_stats.json (optional)."
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--split", type=str, default=None, help="Data split (default: all splits).")
    parser.add_argument("--workers", type=int, default=-1, help="Number of workers (-1 = all cores).")
    parser.add_argument("--vocab_size", type=int, default=None, help="Max vocab size for NTP labels.")
    parser.add_argument(
        "--row_group_size", type=int, default=32768, help="Row group size for parquet output."
    )
    args = parser.parse_args()

    logger = setup_logging(child_name="pretokenize_embedding")

    dataset_path = Path(args.dataset_path)

    # Validate embedding_text exists in MEDS Reader
    emb_text_dir = dataset_path / "embedding_text"
    if not emb_text_dir.is_dir():
        logger.warning(
            f"embedding_text directory not found at {emb_text_dir}. "
            "Ensure MEDS data was generated with generate_embedding_text: true."
        )

    samples_path = args.samples_path or str(dataset_path / "metadata" / "samples.parquet")
    dataset_config = {
        "dataset_path": str(dataset_path),
        "samples_path": samples_path,
        "split": args.split,
    }
    dataset = create_dataset(dataset_config)
    logger.info(f"Dataset: {len(dataset)} samples")

    os.makedirs(args.out_dir, exist_ok=True)
    effective_num_workers = cpu_count() if args.workers == -1 else args.workers

    process_fn = partial(_process_row, vocab_size=args.vocab_size)

    init_args = (
        args.vocab_path,
        args.embedding_lookup_path,
        args.numeric_stats_path,
        args.vocab_size,
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
    output_file = "patients_tokenized.parquet"

    with tqdm(total=len(dataset), desc="pretokenize_embedding") as bar:
        for out in results:
            bar.update()
            if out is None:
                continue
            batch.append(out)
            written += 1
            if len(batch) >= args.row_group_size:
                table = pa.Table.from_pylist(batch, schema=_schema())
                if writer is None:
                    output_path = os.path.join(args.out_dir, output_file)
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                batch = []

    if batch:
        table = pa.Table.from_pylist(batch, schema=_schema())
        if writer is None:
            output_path = os.path.join(args.out_dir, output_file)
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
    if writer:
        writer.close()

    if pool:
        pool.close()
        pool.join()

    logger.info(f"Wrote {written} samples to {args.out_dir}/{output_file}")


if __name__ == "__main__":
    main()
