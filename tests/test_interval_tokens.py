"""
Unit tests for interval token handling in flat tokenization.

Tests cover:
1. Flat interval token insertion and masking
2. Age alignment with interval tokens
3. Collation preserves interval tokens

All tests use realistic synthetic data with actual interval token IDs.
"""

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ehr_fm.data.dataset import TokenizedDataset
from ehr_fm.models.transformer import packed_ehr_collate

# -----------------------------------------------------------------------------
# Test 2.1: Flat Interval Tokens
# -----------------------------------------------------------------------------


class TestFlatIntervalTokens:
    """Test interval token handling in flat tokenization."""

    @pytest.fixture
    def flat_dataset_with_intervals(self, tmp_path):
        """Create flat dataset with interval tokens.

        Assumes interval tokens have low IDs (0-20) as they're typically
        prepended to vocabulary during vocab creation.
        """
        path = tmp_path / "flat_intervals.parquet"

        patients = []

        # Patient 1: Regular tokens + interval tokens in vocab
        # Realistic: INT_1m_5m=5, INT_5m_15m=6, regular codes=100+
        patients.append(
            {
                "subject_id": 1,
                "index_time": datetime(2023, 1, 1),
                "token_ids": [0, 5, 100, 6, 200, 300],  # MEDS_BIRTH, INT_1m_5m, code, INT_5m_15m, codes
                "age": [0.0, 0.001, 0.002, 0.01, 0.011, 0.012],  # Ages in days
                "length": 6,
            }
        )

        # Patient 2: Interval tokens that are OOV (>= num_ntp_classes)
        # If vocab_size=26630 but num_ntp_classes=8192, intervals 8195+ are OOV
        patients.append(
            {
                "subject_id": 2,
                "index_time": datetime(2023, 1, 2),
                "token_ids": [0, 8195, 100, 8196, 200],  # INT tokens as OOV
                "age": [0.0, 0.001, 0.002, 0.01, 0.011],
                "length": 5,
            }
        )

        # Patient 3: No interval tokens (control)
        patients.append(
            {
                "subject_id": 3,
                "index_time": datetime(2023, 1, 3),
                "token_ids": [0, 100, 200, 300],
                "age": [0.0, 1.0, 2.0, 3.0],
                "length": 4,
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

        rows = []
        for p in patients:
            p["age_normalized"] = [0.0] * p["length"]
            rows.append(
                {
                    "subject_id": p["subject_id"],
                    "index_time": p["index_time"],
                    "token_ids": pa.array(p["token_ids"], type=pa.int32()),
                    "age": pa.array(p["age"], type=pa.float32()),
                    "length": p["length"],
                    "age_normalized": pa.array(p["age_normalized"], type=pa.float32()),
                }
            )

        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, compression="zstd")
        return path

    def test_flat_interval_tokens_in_vocab(self, flat_dataset_with_intervals):
        """Test that interval tokens in vocab are included in labels."""
        dataset = TokenizedDataset(
            flat_dataset_with_intervals,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[0]  # Patient 1 with in-vocab intervals

        # Input should be unchanged
        assert torch.equal(sample["input_ids"], torch.tensor([0, 5, 100, 6, 200, 300], dtype=torch.long))

        # Labels: next tokens, last is -100
        # Position 0 (MEDS_BIRTH=0) predicts 5 (INT_1m_5m, in vocab)
        # Position 1 (INT_1m_5m=5) predicts 100 (code, in vocab)
        # etc.
        expected_labels = torch.tensor([5, 100, 6, 200, 300, -100], dtype=torch.long)
        assert torch.equal(sample["labels"], expected_labels)

    def test_flat_interval_tokens_masked_if_oov(self, flat_dataset_with_intervals):
        """Test that interval tokens >= num_ntp_classes are masked in labels."""
        dataset = TokenizedDataset(
            flat_dataset_with_intervals,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[1]  # Patient 2 with OOV intervals

        # Input unchanged
        assert torch.equal(sample["input_ids"], torch.tensor([0, 8195, 100, 8196, 200], dtype=torch.long))

        # Labels: OOV interval tokens (8195, 8196) should be masked
        # Position 0 predicts 8195 (OOV) -> -100
        # Position 1 predicts 100 (in vocab) -> 100
        # Position 2 predicts 8196 (OOV) -> -100
        # Position 3 predicts 200 (in vocab) -> 200
        # Position 4 predicts nothing -> -100
        expected_labels = torch.tensor([-100, 100, -100, 200, -100], dtype=torch.long)
        assert torch.equal(sample["labels"], expected_labels)

    def test_flat_interval_tokens_age_alignment(self, flat_dataset_with_intervals):
        """Test that age array aligns with interval token insertions."""
        dataset = TokenizedDataset(
            flat_dataset_with_intervals,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        sample = dataset[0]

        # Age array should match token count
        assert len(sample["age"]) == len(sample["input_ids"])
        assert len(sample["age"]) == 6

        # Ages should be in order (possibly with small intervals)
        assert torch.all(sample["age"][1:] >= sample["age"][:-1])

    def test_flat_interval_tokens_collation(self, flat_dataset_with_intervals):
        """Test that interval tokens are correctly handled in batch collation."""
        dataset = TokenizedDataset(
            flat_dataset_with_intervals,
            max_length=512,
            one_window=True,
            num_ntp_classes=8192,
        )

        # Batch with all patients
        batch = [dataset[i] for i in range(3)]
        collated = packed_ehr_collate(batch)

        # All tokens should be concatenated (6 + 5 + 4 = 15)
        assert len(collated["input_ids"]) == 15
        assert len(collated["ages"]) == 15

        # Patient lengths should be correct
        assert torch.equal(collated["patient_lengths"], torch.tensor([6, 5, 4], dtype=torch.int32))

        # Label indices should exclude masked positions
        # Patient 1: 5 valid labels (last masked)
        # Patient 2: 2 valid labels (positions 1 and 3, rest masked)
        # Patient 3: 3 valid labels (last masked)
        # Total: 5 + 2 + 3 = 10 valid labels
        assert len(collated["label_indices"]) == 10
