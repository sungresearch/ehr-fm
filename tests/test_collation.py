"""
Unit tests for collation and batch processing.

Tests cover:
1. packed_ehr_collate for flat tokenization
2. Token indices adjustment across batch
3. Label indices extraction
4. TokenBudgetBatchSampler behavior

All tests use realistic synthetic data and test actual implementations.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ehr_fm.data.dataset import TokenizedDataset
from ehr_fm.data.sampler import TokenBudgetBatchSampler
from ehr_fm.models.transformer import packed_ehr_collate

# -----------------------------------------------------------------------------
# Helper functions for creating test data
# -----------------------------------------------------------------------------


def create_flat_parquet(path: Path, patients: list[dict]) -> None:
    """Create flat pretokenized parquet file from patient data."""
    rows = []
    for p in patients:
        rows.append(
            {
                "subject_id": p["subject_id"],
                "index_time": p["index_time"],
                "token_ids": pa.array(p["token_ids"], type=pa.int32()),
                "age": pa.array(p["age"], type=pa.float32()),
                "length": p["length"],
                "age_normalized": pa.array(p.get("age_normalized", [0.0] * p["length"]), type=pa.float32()),
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


# -----------------------------------------------------------------------------
# Test 2.1: Flat Collation
# -----------------------------------------------------------------------------


class TestFlatCollation:
    """Test packed_ehr_collate for flat tokenization."""

    @pytest.fixture
    def flat_dataset_varied_lengths(self, tmp_path):
        """Create flat dataset with varied sequence lengths."""
        path = tmp_path / "flat_varied.parquet"

        patients = [
            {
                "subject_id": 1,
                "index_time": datetime(2023, 1, 1),
                "token_ids": [0, 1, 2, 3, 4],
                "age": [0.0, 1.0, 2.0, 3.0, 4.0],
                "length": 5,
            },
            {
                "subject_id": 2,
                "index_time": datetime(2023, 1, 2),
                "token_ids": [0, 10, 20],
                "age": [0.0, 5.0, 10.0],
                "length": 3,
            },
            {
                "subject_id": 3,
                "index_time": datetime(2023, 1, 3),
                "token_ids": [0, 100, 200, 300, 400, 500, 600],
                "age": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "length": 7,
            },
        ]

        create_flat_parquet(path, patients)
        return path

    def test_flat_token_concatenation(self, flat_dataset_varied_lengths):
        """Test that flat tokens are correctly concatenated across batch."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Verify all tokens are concatenated
        expected_tokens = torch.tensor(
            [0, 1, 2, 3, 4, 0, 10, 20, 0, 100, 200, 300, 400, 500, 600], dtype=torch.long
        )
        assert torch.equal(collated["input_ids"], expected_tokens)

    def test_flat_age_concatenation(self, flat_dataset_varied_lengths):
        """Test that ages are correctly concatenated."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Verify ages
        expected_ages = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32
        )
        assert torch.allclose(collated["ages"], expected_ages)

    def test_flat_patient_lengths(self, flat_dataset_varied_lengths):
        """Test that patient_lengths are correctly recorded."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Verify patient lengths
        expected_lengths = torch.tensor([5, 3, 7], dtype=torch.int32)
        assert torch.equal(collated["patient_lengths"], expected_lengths)

    def test_flat_label_indices_extraction(self, flat_dataset_varied_lengths):
        """Test that label_indices correctly identify non-masked positions."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Patient 1: 5 tokens, last is -100, so 4 valid labels at positions 0-3
        # Patient 2: 3 tokens, last is -100, so 2 valid labels at positions 5-6
        # Patient 3: 7 tokens, last is -100, so 6 valid labels at positions 8-13
        expected_label_indices = torch.tensor([0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13], dtype=torch.long)
        assert torch.equal(collated["label_indices"], expected_label_indices)

    def test_flat_labels_aligned_with_indices(self, flat_dataset_varied_lengths):
        """Test that task labels correspond to label_indices."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Verify labels match the tokens at label_indices positions
        all_labels_flat = torch.cat([batch[i]["labels"] for i in range(3)])
        extracted_labels = all_labels_flat[collated["label_indices"]]

        assert torch.equal(collated["task"]["labels"], extracted_labels)
        assert torch.all(collated["task"]["labels"] != -100)

    def test_flat_patient_ids_and_timestamps(self, flat_dataset_varied_lengths):
        """Test that patient IDs and timestamps are preserved."""
        dataset = TokenizedDataset(
            flat_dataset_varied_lengths,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # Verify patient IDs
        expected_pids = torch.tensor([1, 2, 3], dtype=torch.int64)
        assert torch.equal(collated["patient_ids"], expected_pids)

        # Verify timestamps (as Unix timestamps)
        assert collated["index_times"].shape == (3,)
        assert collated["index_times"].dtype == torch.int64


# -----------------------------------------------------------------------------
# Test 2.2: Token Budget Batch Sampler
# -----------------------------------------------------------------------------


class TestTokenBudgetBatchSampler:
    """Test TokenBudgetBatchSampler for token-based batching."""

    @pytest.fixture
    def dataset_for_sampling(self, tmp_path):
        """Create dataset with known token counts for sampling tests."""
        path = tmp_path / "sampling_test.parquet"

        # Create patients with predictable token counts
        patients = []
        for i in range(20):
            length = (i % 5) + 1  # Lengths: 1, 2, 3, 4, 5, 1, 2, ...
            patients.append(
                {
                    "subject_id": i,
                    "index_time": datetime(2023, 1, 1) + timedelta(days=i),
                    "token_ids": list(range(length)),
                    "age": [float(j) for j in range(length)],
                    "length": length,
                }
            )

        create_flat_parquet(path, patients)
        return path

    def test_batch_token_budget_actually_respected(self, dataset_for_sampling):
        """Test that batches respect the token budget constraint with valid overflow cases."""
        dataset = TokenizedDataset(
            dataset_for_sampling,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        max_tokens = 10
        min_patients = 2
        sampler = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=max_tokens,
            min_patients=min_patients,
            shuffle=False,
            seed=42,
        )

        # Track valid overflows
        overflows = []

        for batch_indices in sampler:
            total_tokens = sum(dataset[i]["length"] for i in batch_indices)

            if total_tokens > max_tokens:
                # Overflow is valid if:
                # 1. Single sample exceeds budget, OR
                # 2. Batch size < min_patients (still building batch)
                max_sample_length = max(dataset[i]["length"] for i in batch_indices)
                is_valid_overflow = (
                    max_sample_length > max_tokens
                    or len(batch_indices)  # Single sample too large
                    < min_patients  # Still building to min_patients
                )
                overflows.append((total_tokens, max_sample_length, len(batch_indices), is_valid_overflow))

                # All overflows should be justified
                assert is_valid_overflow, (
                    f"Invalid overflow: {total_tokens} tokens with max_sample={max_sample_length}, "
                    f"batch_size={len(batch_indices)}, min_patients={min_patients}"
                )
            else:
                # Within budget
                assert total_tokens <= max_tokens

        # Most batches should be within budget
        total_batches = len(
            list(
                TokenBudgetBatchSampler(
                    dataset=dataset,
                    tokens_per_batch=max_tokens,
                    min_patients=min_patients,
                    shuffle=False,
                    seed=42,
                )
            )
        )
        overflow_ratio = len(overflows) / max(total_batches, 1)

        # Allow some overflows but not excessive (< 50% seems reasonable)
        assert overflow_ratio < 0.5, f"Too many overflows: {len(overflows)}/{total_batches}"

    def test_all_samples_covered(self, dataset_for_sampling):
        """Test that all samples appear exactly once per epoch."""
        dataset = TokenizedDataset(
            dataset_for_sampling,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sampler = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=10,
            min_patients=1,
            shuffle=False,
        )

        all_indices = []
        for batch_indices in sampler:
            all_indices.extend(batch_indices)

        # All samples should appear exactly once
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_shuffle_produces_different_orders(self, dataset_for_sampling):
        """Test that shuffle=True produces different batch orders."""
        dataset = TokenizedDataset(
            dataset_for_sampling,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sampler1 = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=10,
            min_patients=1,
            shuffle=True,
            seed=42,
        )

        sampler2 = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=10,
            min_patients=1,
            shuffle=True,
            seed=43,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # Different seeds should produce different orders
        assert batches1 != batches2

    def test_same_seed_produces_same_order(self, dataset_for_sampling):
        """Test that same seed produces reproducible batches."""
        dataset = TokenizedDataset(
            dataset_for_sampling,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sampler1 = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=10,
            min_patients=1,
            shuffle=True,
            seed=42,
        )

        sampler2 = TokenBudgetBatchSampler(
            dataset=dataset,
            tokens_per_batch=10,
            min_patients=1,
            shuffle=True,
            seed=42,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # Same seed should produce same order
        assert batches1 == batches2
