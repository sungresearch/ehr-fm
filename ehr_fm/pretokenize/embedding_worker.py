"""Pool worker for *embedding*-mode pretokenization.

'Embedding' here is the model input mode (each event -> embedding_text_id + NTP
label + numeric feature vector), NOT ehr_fm.embedding, which builds the
text-embedding table. Mirrors ehr_fm.pretokenize.worker: per-process state is
bundled into a single _EmbeddingWorkerState built once by _init_worker (the Pool
initializer, or the sequential path in the driver) and read by _process_row. Both
live in this module so they share one module-global (_worker_state) per worker.
"""

import dataclasses
import datetime
import json
from pathlib import Path

import pyarrow as pa

from ehr_fm.io import read_json_yaml
from ehr_fm.pretokenize.embedding_numeric import (
    _compute_numeric_features_ref_range_priority,
)
from ehr_fm.pretokenize.lookups import _build_token_string_lookup
from ehr_fm.tokenization import JointConfig, JointPolicy, TokenizationPolicy


@dataclasses.dataclass(frozen=True)
class _EmbeddingWorkerState:
    """Immutable per-process state for embedding-mode pretokenization."""

    policy: TokenizationPolicy
    token_lookup: dict[str, int]
    embedding_text_to_id: dict[str, int]
    age_mean: float
    age_std: float
    vocab_size: int


_worker_state: "_EmbeddingWorkerState | None" = None


def _init_worker(
    vocab_path,
    embedding_lookup_path,
    vocab_size,
):
    global _worker_state

    vocab_data = read_json_yaml(vocab_path)
    vocab_entries = vocab_data["vocab"]
    age_stats = vocab_data["age_stats"]
    vocab_config = vocab_data.get("config", {})

    quantile_breaks = vocab_data.get("quantile_breaks", {})
    known_stages = set(vocab_data.get("discovered_stages", []))

    # Build token string → ID lookup (shared with the discrete pipeline)
    token_lookup = _build_token_string_lookup(vocab_entries)

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

    effective_vocab_size = vocab_size if vocab_size else len(vocab_entries)

    _worker_state = _EmbeddingWorkerState(
        policy=policy,
        token_lookup=token_lookup,
        embedding_text_to_id=embedding_text_to_id,
        age_mean=age_stats["mean"],
        age_std=age_stats["std"],
        vocab_size=effective_vocab_size,
    )


def _process_row(row, *, vocab_size):
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

    policy = _worker_state.policy
    token_lookup = _worker_state.token_lookup
    embedding_text_to_id = _worker_state.embedding_text_to_id
    age_mean = _worker_state.age_mean
    age_std = _worker_state.age_std
    effective_vocab_size = vocab_size or _worker_state.vocab_size

    emb_ids = []
    token_ids = []
    numeric_features = []
    ages = []
    ages_normalized = []

    for event in seq:
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

        num_feat = _compute_numeric_features_ref_range_priority(event)

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
