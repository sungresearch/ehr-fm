from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch

from ehr_fm.embedding.build import (
    assign_ids,
    compute_string_set_hash,
    encode_strings,
    extract_unique_strings,
    serialize_artifacts,
)
from ehr_fm.embedding.lookup import EmbeddingLookup


@pytest.fixture
def meds_with_embedding_text(tmp_path):
    """Create a minimal MEDS dataset with embedding_text across two shards."""
    base = tmp_path / "meds_dataset"
    data_dir = base / "data"
    data_dir.mkdir(parents=True)

    birth_time = datetime(2000, 1, 1)

    shard_0 = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                birth_time,
                birth_time + timedelta(days=30),
                birth_time + timedelta(days=60),
            ],
            "code": ["MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/flu"],
            "numeric_value": [None, 5.0, None],
            "embedding_text": [
                "Birth event",
                "Lab: Glucose. Value: 5.0 mmol/L",
                "Diagnosis: Influenza",
            ],
        }
    )
    for col, dtype in shard_0.schema.items():
        if dtype == pl.Null:
            shard_0 = shard_0.with_columns(pl.col(col).cast(pl.Float64))
    shard_0.write_parquet(data_dir / "data_0.parquet")

    shard_1 = pl.DataFrame(
        {
            "subject_id": [2, 2],
            "time": [birth_time, birth_time + timedelta(days=10)],
            "code": ["MEDS_BIRTH", "LAB/glucose"],
            "numeric_value": [None, 7.2],
            "embedding_text": [
                "Birth event",
                "Lab: Glucose. Value: 7.2 mmol/L",
            ],
        }
    )
    for col, dtype in shard_1.schema.items():
        if dtype == pl.Null:
            shard_1 = shard_1.with_columns(pl.col(col).cast(pl.Float64))
    shard_1.write_parquet(data_dir / "data_1.parquet")

    return base


@pytest.fixture
def sample_artifacts(tmp_path):
    """Create a minimal set of embedding lookup artifacts on disk."""
    artifact_dir = tmp_path / "artifacts"
    strings = ["Alpha", "Beta", "Gamma"]
    embedding_dim = 8
    table = np.random.default_rng(42).standard_normal((3, embedding_dim)).astype(np.float16)
    id_mapping = {s: i for i, s in enumerate(strings)}
    metadata = {
        "model_name": "test-model",
        "string_set_hash": "abc123",
        "n_embeddings": 3,
        "embedding_dim": embedding_dim,
        "dtype": "float16",
        "creation_timestamp": "2026-01-01T00:00:00Z",
    }
    serialize_artifacts(artifact_dir, table, id_mapping, metadata)
    return artifact_dir, table, id_mapping, metadata


class TestExtractUniqueStrings:
    def test_returns_unique_across_shards(self, meds_with_embedding_text):
        result = extract_unique_strings(meds_with_embedding_text)
        assert set(result) == {
            "Birth event",
            "Lab: Glucose. Value: 5.0 mmol/L",
            "Diagnosis: Influenza",
            "Lab: Glucose. Value: 7.2 mmol/L",
        }

    def test_deduplicates_shared_strings(self, meds_with_embedding_text):
        result = extract_unique_strings(meds_with_embedding_text)
        assert result.count("Birth event") == 1


class TestAssignIds:
    def test_lexicographic_order(self):
        strings = ["cherry", "apple", "banana"]
        sorted_strings, id_mapping = assign_ids(strings)
        assert sorted_strings == ["apple", "banana", "cherry"]
        assert id_mapping == {"apple": 0, "banana": 1, "cherry": 2}

    def test_deterministic(self):
        strings_a = ["z", "a", "m"]
        strings_b = ["m", "z", "a"]
        _, map_a = assign_ids(strings_a)
        _, map_b = assign_ids(strings_b)
        assert map_a == map_b

    def test_ids_contiguous(self):
        strings = [f"s{i}" for i in range(100)]
        _, id_mapping = assign_ids(strings)
        assert set(id_mapping.values()) == set(range(100))


class TestComputeStringSetHash:
    def test_deterministic(self):
        strings = ["a", "b", "c"]
        assert compute_string_set_hash(strings) == compute_string_set_hash(strings)

    def test_order_sensitive(self):
        assert compute_string_set_hash(["a", "b"]) != compute_string_set_hash(["b", "a"])


class TestEncodeStrings:
    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_calls_model(self, mock_st_class):
        mock_model = MagicMock()
        fake_embeddings = np.random.randn(3, 16).astype(np.float32)
        mock_model.encode.return_value = fake_embeddings
        mock_st_class.return_value = mock_model

        result = encode_strings(["a", "b", "c"], "test-model", batch_size=2, device="cpu")

        mock_st_class.assert_called_once_with("test-model", device="cpu")
        mock_model.encode.assert_called_once()
        assert result.dtype == np.float16
        assert result.shape == (3, 16)
        np.testing.assert_array_almost_equal(result, fake_embeddings.astype(np.float16), decimal=2)


class TestSerializeAndLookup:
    def test_roundtrip(self, sample_artifacts):
        artifact_dir, original_table, original_mapping, original_meta = sample_artifacts

        lookup = EmbeddingLookup(artifact_dir)

        assert lookup.num_embeddings == 3
        assert lookup.embedding_dim == 8
        np.testing.assert_array_equal(lookup.embedding_table, original_table)
        assert lookup.text_to_id == original_mapping
        assert lookup.metadata["model_name"] == "test-model"
        assert lookup.metadata["string_set_hash"] == "abc123"

    def test_get_id(self, sample_artifacts):
        artifact_dir, _, _, _ = sample_artifacts
        lookup = EmbeddingLookup(artifact_dir)

        assert lookup.get_id("Alpha") == 0
        assert lookup.get_id("Beta") == 1
        assert lookup.get_id("Gamma") == 2

    def test_get_id_missing_raises(self, sample_artifacts):
        artifact_dir, _, _, _ = sample_artifacts
        lookup = EmbeddingLookup(artifact_dir)

        with pytest.raises(KeyError):
            lookup.get_id("nonexistent")

    def test_as_torch_embedding_frozen(self, sample_artifacts):
        artifact_dir, original_table, _, _ = sample_artifacts
        lookup = EmbeddingLookup(artifact_dir)

        emb = lookup.as_torch_embedding(freeze=True)

        assert isinstance(emb, torch.nn.Embedding)
        assert emb.weight.shape == (3, 8)
        assert not emb.weight.requires_grad
        expected = torch.from_numpy(original_table.astype(np.float32))
        torch.testing.assert_close(emb.weight.data, expected)

    def test_as_torch_embedding_unfrozen(self, sample_artifacts):
        artifact_dir, _, _, _ = sample_artifacts
        lookup = EmbeddingLookup(artifact_dir)

        emb = lookup.as_torch_embedding(freeze=False)
        assert emb.weight.requires_grad
