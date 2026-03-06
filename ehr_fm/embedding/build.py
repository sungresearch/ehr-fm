import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from safetensors.numpy import save_file

from ehr_fm.logger import setup_logging

from .config import EmbeddingLookupConfig

logger = setup_logging(child_name="embedding.build")


def extract_unique_strings(meds_dir: Path) -> list[str]:
    """Scan all MEDS parquet shards and return unique embedding_text values."""
    pattern = str(meds_dir / "data" / "*.parquet")
    logger.info(f"Scanning parquet shards: {pattern}")

    unique_texts = (
        pl.scan_parquet(pattern)
        .select("embedding_text")
        .drop_nulls()
        .unique()
        .collect()
        .get_column("embedding_text")
        .to_list()
    )

    logger.info(f"Extracted {len(unique_texts)} unique embedding_text strings")
    return unique_texts


def assign_ids(strings: list[str]) -> tuple[list[str], dict[str, int]]:
    """Sort strings lexicographically and assign deterministic integer IDs.

    Returns the sorted list and the string-to-id mapping.
    """
    sorted_strings = sorted(strings)
    id_mapping = {s: i for i, s in enumerate(sorted_strings)}
    return sorted_strings, id_mapping


def compute_string_set_hash(sorted_strings: list[str]) -> str:
    """Compute a deterministic SHA-256 hash over the sorted string set."""
    h = hashlib.sha256()
    for s in sorted_strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_strings(
    sorted_strings: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode strings with a frozen text embedding model.

    Returns an (N, D) float16 numpy array where the i-th row corresponds
    to the string with embedding_text_id = i.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logger.info(
        f"Encoding {len(sorted_strings)} strings "
        f"(batch_size={batch_size}, device={device})"
    )
    embeddings = model.encode(
        sorted_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = embeddings.astype(np.float16)
    logger.info(f"Embedding table shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings


def serialize_artifacts(
    output_dir: Path,
    embedding_table: np.ndarray,
    id_mapping: dict[str, int],
    metadata: dict,
) -> None:
    """Write embedding table, ID mapping, and metadata to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "embedding_table.safetensors"
    save_file({"embedding_table": embedding_table}, str(table_path))
    logger.info(f"Saved embedding table: {table_path}")

    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f)
    logger.info(f"Saved ID mapping ({len(id_mapping)} entries): {mapping_path}")

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")


def build_embedding_lookup(config: EmbeddingLookupConfig) -> None:
    """Run the full embedding lookup construction pipeline."""
    unique_strings = extract_unique_strings(config.meds_dir)

    sorted_strings, id_mapping = assign_ids(unique_strings)
    string_set_hash = compute_string_set_hash(sorted_strings)
    logger.info(f"String set hash: {string_set_hash[:16]}...")

    embedding_table = encode_strings(
        sorted_strings,
        model_name=config.model_name,
        batch_size=config.batch_size,
        device=config.device,
    )

    metadata = {
        "model_name": config.model_name,
        "string_set_hash": string_set_hash,
        "n_embeddings": embedding_table.shape[0],
        "embedding_dim": embedding_table.shape[1],
        "dtype": config.dtype,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    serialize_artifacts(config.output_dir, embedding_table, id_mapping, metadata)
    logger.info(f"Embedding lookup complete: {config.output_dir}")
