"""
Unit tests for dataset label creation and masking.

Tests cover:
1. Flat dataset next token prediction with vocab cutoff

All tests use realistic synthetic data based on actual MEDS datasets.
Tests instantiate actual Dataset classes and test their __getitem__ methods.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ehr_fm.data.dataset import TokenizedDataset

# -----------------------------------------------------------------------------
# Realistic synthetic data generators
# -----------------------------------------------------------------------------


def get_realistic_stats():
    """Return realistic statistics from actual MEDS datasets."""
    return {
        "vocab_size": 26630,
        "num_ntp_classes": 8192,  # Common training cutoff
        "age_stats": {
            "mean": 591451391.4389731,  # Mean age in seconds (~18.7 years)
            "std": 651523637.5471003,  # ~20.6 years in seconds
        },
    }


def create_flat_pretokenized_parquet(
    output_path: Path,
    num_patients: int = 10,
    min_length: int = 2,
    max_length: int = 100,
    vocab_size: int = 26630,
    include_oov: bool = False,
) -> None:
    """Create realistic flat pretokenized parquet file."""
    rows = []
    base_time = datetime(2023, 1, 1)

    for patient_id in range(num_patients):
        length = random.randint(min_length, max_length)

        # Generate tokens
        tokens = []
        for i in range(length):
            if include_oov and random.random() < 0.1:
                # OOV token
                tokens.append(random.randint(vocab_size, vocab_size + 1000))
            else:
                # In-vocab token (power-law distribution)
                if random.random() < 0.7:
                    tokens.append(random.randint(0, vocab_size // 10))
                else:
                    tokens.append(random.randint(0, vocab_size - 1))

        # Realistic ages: days since birth, increasing
        ages = sorted([random.uniform(0, 30000) for _ in range(length)])

        # Age normalized (mean ~591M seconds, std ~651M seconds)
        age_mean = 591451391.4389731
        age_std = 651523637.5471003
        ages_normalized = [(a * 86400 - age_mean) / age_std for a in ages]

        rows.append(
            {
                "subject_id": patient_id + 1000000,
                "index_time": base_time + timedelta(days=patient_id * 30),
                "token_ids": pa.array(tokens, type=pa.int32()),
                "age": pa.array(ages, type=pa.float32()),
                "length": length,
                "age_normalized": pa.array(ages_normalized, type=pa.float32()),
            }
        )

    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
            pa.field("age_normalized", pa.list_(pa.float32())),
        ]
    )

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path, compression="zstd")


# -----------------------------------------------------------------------------
# Test 1.1: Flat Dataset Next Token Prediction
# -----------------------------------------------------------------------------


class TestFlatDatasetLabelCreation:
    """Test TokenizedDataset label creation and masking using actual dataset class."""

    @pytest.fixture
    def flat_dataset_small(self, tmp_path):
        """Create small flat dataset with known tokens."""
        path = tmp_path / "flat_small.parquet"

        # Manually create controlled data
        rows = []
        # Patient 1: all in-vocab tokens
        rows.append(
            {
                "subject_id": 1,
                "index_time": datetime(2023, 1, 1),
                "token_ids": pa.array([0, 1, 3, 10, 25], type=pa.int32()),
                "age": pa.array([0.0, 1.0, 2.0, 3.0, 4.0], type=pa.float32()),
                "length": 5,
                "age_normalized": pa.array([0.0] * 5, type=pa.float32()),
            }
        )

        # Patient 2: contains OOV tokens (>= 8192)
        rows.append(
            {
                "subject_id": 2,
                "index_time": datetime(2023, 1, 2),
                "token_ids": pa.array([0, 50, 8200, 100, 8500], type=pa.int32()),
                "age": pa.array([0.0, 1.0, 2.0, 3.0, 4.0], type=pa.float32()),
                "length": 5,
                "age_normalized": pa.array([0.0] * 5, type=pa.float32()),
            }
        )

        # Patient 3: single token
        rows.append(
            {
                "subject_id": 3,
                "index_time": datetime(2023, 1, 3),
                "token_ids": pa.array([0], type=pa.int32()),
                "age": pa.array([0.0], type=pa.float32()),
                "length": 1,
                "age_normalized": pa.array([0.0], type=pa.float32()),
            }
        )

        schema = pa.schema(
            [
                pa.field("subject_id", pa.int64()),
                pa.field("index_time", pa.timestamp("ns")),
                pa.field("token_ids", pa.list_(pa.int32())),
                pa.field("age", pa.list_(pa.float32())),
                pa.field("length", pa.int32()),
                pa.field("age_normalized", pa.list_(pa.float32())),
            ]
        )

        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, compression="zstd")
        return path

    @pytest.fixture
    def flat_dataset_realistic(self, tmp_path):
        """Create realistic flat dataset."""
        path = tmp_path / "flat_realistic.parquet"
        create_flat_pretokenized_parquet(
            path,
            num_patients=10,
            min_length=5,
            max_length=100,
            vocab_size=26630,
            include_oov=True,
        )
        return path

    def test_basic_label_shift(self, flat_dataset_small):
        """Test that labels are input tokens shifted left by 1."""
        dataset = TokenizedDataset(
            flat_dataset_small,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[0]  # Patient 1: [0, 1, 3, 10, 25]

        # Verify labels are shifted inputs
        assert torch.equal(sample["input_ids"], torch.tensor([0, 1, 3, 10, 25], dtype=torch.long))
        assert torch.equal(sample["labels"], torch.tensor([1, 3, 10, 25, -100], dtype=torch.long))

    def test_vocab_cutoff_masking(self, flat_dataset_small):
        """Test that tokens >= num_ntp_classes are masked."""
        dataset = TokenizedDataset(
            flat_dataset_small,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[1]  # Patient 2: [0, 50, 8200, 100, 8500]

        # Verify input_ids unchanged
        assert torch.equal(sample["input_ids"], torch.tensor([0, 50, 8200, 100, 8500], dtype=torch.long))

        # Verify OOV targets masked: [50, -100 (8200 OOV), 100, -100 (8500 OOV), -100 (final)]
        assert torch.equal(sample["labels"], torch.tensor([50, -100, 100, -100, -100], dtype=torch.long))

    def test_single_token_sequence(self, flat_dataset_small):
        """Test edge case: single token (MEDS_BIRTH only)."""
        dataset = TokenizedDataset(
            flat_dataset_small,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[2]  # Patient 3: [0]

        # Single token has no next token to predict
        assert torch.equal(sample["input_ids"], torch.tensor([0], dtype=torch.long))
        assert torch.equal(sample["labels"], torch.tensor([-100], dtype=torch.long))

    def test_realistic_sequence(self, flat_dataset_realistic):
        """Test with realistic token distribution."""
        dataset = TokenizedDataset(
            flat_dataset_realistic,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        for i in range(min(5, len(dataset))):
            sample = dataset[i]

            # Verify types and shapes
            assert isinstance(sample["input_ids"], torch.Tensor)
            assert isinstance(sample["labels"], torch.Tensor)
            assert sample["input_ids"].dtype == torch.long
            assert sample["labels"].dtype == torch.long
            assert len(sample["input_ids"]) == len(sample["labels"])

            # Verify last label is always -100
            assert sample["labels"][-1] == -100

            # Verify all other labels are correctly shifted or masked
            for j in range(len(sample["labels"]) - 1):
                next_token = sample["input_ids"][j + 1].item()
                label = sample["labels"][j].item()

                if label == -100:
                    # Must be OOV
                    assert next_token >= 8192
                else:
                    # Must match next token
                    assert label == next_token
                    assert next_token < 8192


# Flat collation tests are in test_collation.py
