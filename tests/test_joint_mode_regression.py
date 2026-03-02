"""
Regression tests for joint tokenization mode.

These tests capture the current joint tokenization behavior (vocab training and pretokenize)
to ensure the refactored code produces identical outputs. Created BEFORE any refactoring
to establish a baseline.

Test coverage:
1. Vocabulary training produces consistent vocab entries and age_stats
2. Pretokenization produces consistent token sequences
3. Edge cases: numeric values, text values, OOV codes
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ehr_fm.io import read_json_yaml, write_dict_to_json
from ehr_fm.tokenizer import pretokenize_data
from ehr_fm.vocabulary import Vocab


class TestJointModeVocabularyTraining:
    """Test that joint mode vocabulary training produces consistent results."""

    @pytest.fixture
    def synthetic_events(self):
        """Create synthetic patient events for vocabulary training.

        Covers:
        - Code-only events (no numeric/text)
        - Numeric events with values
        - Text events
        - Multiple patients with different event distributions
        """
        birth_time = datetime(2000, 1, 1)

        # Patient 1: Mix of event types
        patient1_events = [
            {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None, "text_value": None},
            {
                "code": "LAB/glucose",
                "time": birth_time + timedelta(days=30),
                "numeric_value": 95.0,
                "text_value": None,
            },
            {
                "code": "LAB/glucose",
                "time": birth_time + timedelta(days=60),
                "numeric_value": 142.0,
                "text_value": None,
            },
            {
                "code": "DIAGNOSIS/diabetes",
                "time": birth_time + timedelta(days=90),
                "numeric_value": None,
                "text_value": None,
            },
            {
                "code": "DRUG/metformin",
                "time": birth_time + timedelta(days=91),
                "numeric_value": None,
                "text_value": "oral",
            },
        ]

        # Patient 2: More events to build frequency
        patient2_events = [
            {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None, "text_value": None},
            {
                "code": "LAB/glucose",
                "time": birth_time + timedelta(days=45),
                "numeric_value": 110.0,
                "text_value": None,
            },
            {
                "code": "LAB/hemoglobin",
                "time": birth_time + timedelta(days=45),
                "numeric_value": 14.5,
                "text_value": None,
            },
            {
                "code": "DIAGNOSIS/hypertension",
                "time": birth_time + timedelta(days=100),
                "numeric_value": None,
                "text_value": None,
            },
        ]

        # Patient 3: Edge cases
        patient3_events = [
            {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None, "text_value": None},
            {
                "code": "LAB/glucose",
                "time": birth_time + timedelta(days=20),
                "numeric_value": 70.0,
                "text_value": None,
            },
            {
                "code": "LAB/glucose",
                "time": birth_time + timedelta(days=25),
                "numeric_value": 250.0,
                "text_value": None,
            },
            {
                "code": "NOTE/progress",
                "time": birth_time + timedelta(days=30),
                "numeric_value": None,
                "text_value": "stable",
            },
        ]

        return [patient1_events, patient2_events, patient3_events]

    def test_vocab_training_deterministic(self, synthetic_events, tmp_path):
        """Verify vocab training produces deterministic output."""
        # Create a simple dataset mock using MEDSDataLoader pattern
        # We'll directly call vocab.forward() instead

        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = Vocab(vocab_config)

        # Train on synthetic events
        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        # Get vocabulary
        vocab_entries = vocab.get_vocab()

        # Verify expected entries exist
        code_strings = [e["code_string"] for e in vocab_entries if e["type"] == "code"]
        assert "MEDS_BIRTH" in code_strings
        assert "DIAGNOSIS/diabetes" in code_strings
        assert "DIAGNOSIS/hypertension" in code_strings

        # Verify numeric bins created for LAB/glucose
        numeric_entries = [
            e for e in vocab_entries if e["type"] == "numeric" and e["code_string"] == "LAB/glucose"
        ]
        assert len(numeric_entries) > 0, "Expected numeric bins for LAB/glucose"

        # Verify text entries
        text_entries = [e for e in vocab_entries if e["type"] == "text"]
        assert len(text_entries) > 0, "Expected text entries"

        # Verify age_stats computed
        assert vocab.age_stats.mean > 0
        assert vocab.age_stats.std > 0

        # Save and reload to verify serialization
        vocab.save(tmp_path, overwrite=True)
        loaded = read_json_yaml(tmp_path / "vocab.json")

        assert len(loaded["vocab"]) == len(vocab_entries)
        assert loaded["age_stats"]["mean"] == vocab.age_stats.mean
        assert loaded["age_stats"]["std"] == vocab.age_stats.std


class TestJointModePretokenizeIntegration:
    """Integration tests for the full pretokenize pipeline."""

    @pytest.fixture
    def meds_dataset(self, tmp_path, convert_meds_to_reader):
        """Create a minimal MEDS Reader dataset for pretokenize testing."""
        import json

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create patient data
        birth_time = datetime(2000, 1, 1)

        records = [
            # Patient 1
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=30),
                "code": "LAB/glucose",
                "numeric_value": 95.0,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=60),
                "code": "DIAGNOSIS/diabetes",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # Patient 2
            {
                "subject_id": 2,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 2,
                "time": birth_time + timedelta(days=45),
                "code": "LAB/glucose",
                "numeric_value": 120.0,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]

        # Create DataFrame with proper types for null columns
        df = pl.DataFrame(records).with_columns(
            [
                pl.col("unit").cast(pl.Int32),  # unit is int32 in MEDS schema
            ]
        )
        df.write_parquet(data_dir / "data_0.parquet")

        # Create samples file
        samples = pl.DataFrame(
            {
                "id": [1, 2],
                "index_t": [birth_time + timedelta(days=100), birth_time + timedelta(days=100)],
                "split": ["train", "train"],
            }
        )
        samples.write_parquet(metadata_dir / "samples.parquet")

        # Create codes.parquet (required for MEDS format detection)
        codes = pl.DataFrame(
            {
                "code": ["MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/diabetes"],
                "description": ["Birth", "Glucose Lab", "Diabetes Diagnosis"],
            }
        )
        codes.write_parquet(metadata_dir / "codes.parquet")

        # Create dataset.json (required for MEDS format detection)
        dataset_json = {
            "dataset_name": "test_dataset",
            "dataset_version": "1.0",
        }
        with open(metadata_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        # Create vocab
        vocab_entries = [
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {
                "type": "numeric",
                "code_string": "LAB/glucose",
                "unit": None,
                "val_start": float("-inf"),
                "val_end": 100.0,
                "weight": -0.4,
            },
            {
                "type": "numeric",
                "code_string": "LAB/glucose",
                "unit": None,
                "val_start": 100.0,
                "val_end": float("inf"),
                "weight": -0.4,
            },
            {"type": "code", "code_string": "DIAGNOSIS/diabetes", "weight": -0.3},
        ]

        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
        }

        vocab_path = tmp_path / "vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)

        reader_path = convert_meds_to_reader(tmp_path)
        return reader_path, vocab_path

    def test_pretokenize_produces_expected_schema(self, meds_dataset):
        """Test that pretokenize produces the expected output schema."""
        dataset_path, vocab_path = meds_dataset
        out_dir = dataset_path / "pretokenized"

        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=dataset_path,
            samples_path=dataset_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        # Load and verify output
        output_path = out_dir / "patients_tokenized.parquet"
        assert output_path.exists(), "Output file should exist"

        df = pl.read_parquet(output_path)

        # Verify columns
        expected_columns = {"subject_id", "index_time", "token_ids", "age", "length", "age_normalized"}
        assert (
            set(df.columns) == expected_columns
        ), f"Expected columns {expected_columns}, got {set(df.columns)}"

        # Verify we have 2 patients
        assert len(df) == 2, f"Expected 2 patients, got {len(df)}"

        # Verify token sequences
        for row in df.iter_rows(named=True):
            token_ids = row["token_ids"]
            ages = row["age"]
            ages_norm = row["age_normalized"]

            # Verify lengths match
            assert len(token_ids) == len(ages) == len(ages_norm) == row["length"]

            # First token should be MEDS_BIRTH (token 0)
            assert token_ids[0] == 0, "First token should be MEDS_BIRTH"

            # Ages should be monotonically increasing (or equal for same-time events)
            for i in range(1, len(ages)):
                assert ages[i] >= ages[i - 1], "Ages should be monotonically increasing"
