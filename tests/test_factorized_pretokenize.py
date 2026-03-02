"""Tests for factorized tokenization mode in pretokenize.

The new API reads tokenization_mode and config from the vocab file itself.
This file creates vocab files with proper config metadata for testing.
"""

import json
from datetime import datetime, timedelta

import polars as pl
import pytest

from ehr_fm.io import write_dict_to_json
from ehr_fm.tokenizer import pretokenize_data


class TestFactorizedPretokenize:
    """Test factorized mode pretokenization.

    Key change in new API: tokenization mode and config are read from vocab file.
    No more explicit tokenization_mode, factorized_config, quantile_breaks, known_stages args.
    """

    @pytest.fixture
    def meds_dataset(self, tmp_path, convert_meds_to_reader):
        """Create MEDS Reader dataset for testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        birth_time = datetime(2000, 1, 1)
        records = [
            # Patient 1: Events with numeric values
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
                "numeric_value": 80.0,  # Should become Q:1
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=60),
                "code": "LAB/glucose",
                "numeric_value": 120.0,  # Should become Q:2
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=90),
                "code": "DIAGNOSIS/diabetes",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # Patient 2: OOV code
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
                "time": birth_time + timedelta(days=30),
                "code": "LAB/unknown",  # OOV code
                "numeric_value": 100.0,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]

        df = pl.DataFrame(records).with_columns([pl.col("unit").cast(pl.Int32)])
        df.write_parquet(data_dir / "data_0.parquet")

        # Create metadata
        codes = pl.DataFrame(
            {
                "code": ["MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/diabetes"],
                "description": ["Birth", "Glucose", "Diabetes"],
            }
        )
        codes.write_parquet(metadata_dir / "codes.parquet")

        dataset_json = {"dataset_name": "test", "dataset_version": "1.0"}
        with open(metadata_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        samples = pl.DataFrame(
            {
                "id": [1, 2],
                "index_t": [birth_time + timedelta(days=100), birth_time + timedelta(days=100)],
                "split": ["train", "train"],
            }
        )
        samples.write_parquet(metadata_dir / "samples.parquet")

        reader_path = convert_meds_to_reader(tmp_path)
        return reader_path

    @pytest.fixture
    def factorized_vocab_path(self, tmp_path):
        """Create vocab file with factorized mode config."""
        # Vocab entries with proper structure for factorized mode:
        # - Q:* tokens for quantiles
        # - STAGE:* tokens
        # - Base code tokens (with prefixes removed)
        vocab_entries = [
            # Quantile tokens (prepended first for low IDs)
            {"type": "quantile", "label": "Q:1", "weight": -1.0},
            {"type": "quantile", "label": "Q:2", "weight": -1.0},
            {"type": "quantile", "label": "Q:3", "weight": -1.0},
            {"type": "quantile", "label": "Q:UNK", "weight": -1.0},
            # Stage tokens
            {"type": "stage", "label": "STAGE:taken", "weight": -1.0},
            {"type": "stage", "label": "STAGE:UNK", "weight": -1.0},
            # Base code tokens (Note: prefixes removed per config)
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "glucose", "weight": -0.4},  # LAB/ prefix removed
            {"type": "code", "code_string": "diabetes", "weight": -0.3},  # DIAGNOSIS/ prefix removed
        ]

        # Vocab file structure with config section
        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
            # Config section - this is what pretokenize_data reads now
            "config": {
                "tokenization_mode": "factorized",
                "emit_quantiles": True,
                "emit_text": False,
                "emit_stage": False,
                "num_quantiles": 10,
                "remove_prefixes": True,
                "separator": "/",
            },
            # Quantile breaks for numeric values
            # LAB/glucose: < 100 → Q:1, 100-150 → Q:2, >= 150 → Q:3
            "quantile_breaks": {"LAB/glucose": [100.0, 150.0]},
            # Discovered stages from training
            "discovered_stages": ["taken"],
        }

        vocab_path = tmp_path / "factorized_vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)
        return vocab_path

    def test_factorized_mode_emits_quantile_tokens(self, meds_dataset, factorized_vocab_path):
        """Test that factorized mode emits Q:* tokens for numeric values."""
        out_dir = meds_dataset / "pretokenized"

        pretokenize_data(
            vocab_path=factorized_vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset,
            samples_path=meds_dataset / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        # Load output
        output_path = out_dir / "patients_tokenized.parquet"
        assert output_path.exists()

        df = pl.read_parquet(output_path)
        assert len(df) == 2  # 2 patients

        # Check patient 1
        patient1 = df.filter(pl.col("subject_id") == 1)
        assert len(patient1) == 1

        tokens = patient1["token_ids"][0].to_list()
        # Expected tokens for patient 1:
        # MEDS_BIRTH (6), glucose (7) + Q:1 (0),
        # glucose (7) + Q:2 (1), diabetes (8)
        # = [6, 7, 0, 7, 1, 8]
        assert 6 in tokens  # MEDS_BIRTH
        assert 7 in tokens  # glucose
        assert 8 in tokens  # diabetes
        assert 0 in tokens  # Q:1
        assert 1 in tokens  # Q:2

    def test_factorized_mode_no_orphans(self, meds_dataset, factorized_vocab_path):
        """Test no-orphans invariant: OOV base code skips all attribute tokens."""
        out_dir = meds_dataset / "pretokenized"

        pretokenize_data(
            vocab_path=factorized_vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset,
            samples_path=meds_dataset / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")

        # Check patient 2 (has OOV code LAB/unknown)
        patient2 = df.filter(pl.col("subject_id") == 2)
        assert len(patient2) == 1

        tokens = patient2["token_ids"][0].to_list()
        # Patient 2 should only have MEDS_BIRTH token
        # LAB/unknown is OOV, so its Q:* token is also skipped (no-orphans)
        assert tokens == [6]  # Only MEDS_BIRTH

    def test_factorized_mode_ages_aligned(self, meds_dataset, factorized_vocab_path):
        """Test that ages are aligned with tokens (same length arrays)."""
        out_dir = meds_dataset / "pretokenized"

        pretokenize_data(
            vocab_path=factorized_vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset,
            samples_path=meds_dataset / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")

        # Check patient 1 ages
        patient1 = df.filter(pl.col("subject_id") == 1)
        tokens = patient1["token_ids"][0].to_list()
        ages = patient1["age"][0].to_list()
        ages_norm = patient1["age_normalized"][0].to_list()

        # All arrays should have same length
        assert len(tokens) == len(ages) == len(ages_norm) == patient1["length"][0]

        # Ages should be monotonically increasing (tokens from same event have same age)
        for i in range(1, len(ages)):
            assert ages[i] >= ages[i - 1]

    def test_joint_mode_backward_compatible(self, meds_dataset, tmp_path):
        """Verify joint mode pretokenize runs without errors."""
        out_dir = meds_dataset / "pretokenized_joint"

        # Create a joint-mode vocab file
        vocab_entries = [
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "glucose", "weight": -0.4},
            {"type": "code", "code_string": "glucose/Q:1", "weight": -0.3},
            {"type": "code", "code_string": "glucose/Q:2", "weight": -0.2},
            {"type": "code", "code_string": "diabetes", "weight": -0.1},
        ]
        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
            "config": {
                "tokenization_mode": "joint",
                "emit_quantiles": True,
                "emit_text": False,
                "emit_stage": False,
                "remove_prefixes": True,
                "separator": "/",
            },
            "quantile_breaks": {"LAB/glucose": [100.0, 150.0]},
            "discovered_stages": [],
        }
        joint_vocab_path = tmp_path / "joint_vocab.json"
        write_dict_to_json(vocab_data, joint_vocab_path, overwrite=True)

        pretokenize_data(
            vocab_path=joint_vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset,
            samples_path=meds_dataset / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        # Verify output is created and has expected structure
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        assert len(df) == 2
        assert "token_ids" in df.columns
        assert "age" in df.columns
        assert "age_normalized" in df.columns

        # Patient 1 should have at least MEDS_BIRTH token
        patient1 = df.filter(pl.col("subject_id") == 1)
        tokens = patient1["token_ids"][0].to_list()
        assert 0 in tokens  # MEDS_BIRTH

    def test_legacy_vocab_no_config(self, meds_dataset, tmp_path):
        """Legacy vocab without config section uses old tokenizer path."""
        out_dir = meds_dataset / "pretokenized_legacy"

        # Create a legacy vocab file (no config section)
        vocab_entries = [
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "LAB/glucose", "weight": -0.4},
            {"type": "code", "code_string": "DIAGNOSIS/diabetes", "weight": -0.3},
        ]
        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
            # No "config" section - legacy vocab
        }
        legacy_vocab_path = tmp_path / "legacy_vocab.json"
        write_dict_to_json(vocab_data, legacy_vocab_path, overwrite=True)

        pretokenize_data(
            vocab_path=legacy_vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset,
            samples_path=meds_dataset / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        assert len(df) == 2

        # Legacy mode uses direct code lookup
        patient1 = df.filter(pl.col("subject_id") == 1)
        tokens = patient1["token_ids"][0].to_list()
        assert 0 in tokens  # MEDS_BIRTH
        # Verify token length is reasonable (at least MEDS_BIRTH + some other tokens)
        assert len(tokens) >= 1


class TestFactorizedModeWithStages:
    """Test factorized mode with workflow_stage tokens."""

    @pytest.fixture
    def meds_dataset_with_stages(self, tmp_path, convert_meds_to_reader):
        """Create MEDS Reader dataset with workflow_stage field."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        birth_time = datetime(2000, 1, 1)
        records = [
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "workflow_stage": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=30),
                "code": "LAB/glucose",
                "numeric_value": 90.0,
                "text_value": None,
                "workflow_stage": "taken",  # Known stage
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=60),
                "code": "LAB/glucose",
                "numeric_value": 100.0,
                "text_value": None,
                "workflow_stage": "unknown_stage",  # Unknown stage → STAGE:UNK
                "description": "",
                "unit": None,
            },
        ]

        df = pl.DataFrame(records).with_columns([pl.col("unit").cast(pl.Int32)])
        df.write_parquet(data_dir / "data_0.parquet")

        codes = pl.DataFrame({"code": ["MEDS_BIRTH", "LAB/glucose"], "description": ["Birth", "Glucose"]})
        codes.write_parquet(metadata_dir / "codes.parquet")

        dataset_json = {"dataset_name": "test", "dataset_version": "1.0"}
        with open(metadata_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        samples = pl.DataFrame({"id": [1], "index_t": [birth_time + timedelta(days=100)], "split": ["train"]})
        samples.write_parquet(metadata_dir / "samples.parquet")

        reader_path = convert_meds_to_reader(tmp_path)
        return reader_path

    def test_stage_tokens_emitted(self, meds_dataset_with_stages, tmp_path):
        """Test that STAGE:* tokens are emitted for workflow_stage."""
        vocab_entries = [
            {"type": "quantile", "label": "Q:1", "weight": -1.0},
            {"type": "quantile", "label": "Q:2", "weight": -1.0},
            {"type": "quantile", "label": "Q:UNK", "weight": -1.0},
            {"type": "stage", "label": "STAGE:taken", "weight": -1.0},
            {"type": "stage", "label": "STAGE:UNK", "weight": -1.0},
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "glucose", "weight": -0.4},
        ]
        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
            "config": {
                "tokenization_mode": "factorized",
                "emit_quantiles": True,
                "emit_text": False,
                "emit_stage": True,
                "remove_prefixes": True,
                "separator": "/",
            },
            "quantile_breaks": {"LAB/glucose": [95.0]},  # 90 → Q:1, 100 → Q:2
            "discovered_stages": ["taken"],
        }
        vocab_path = tmp_path / "vocab_with_stages.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)

        out_dir = tmp_path / "pretokenized"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset_with_stages,
            samples_path=meds_dataset_with_stages / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        tokens = df["token_ids"][0].to_list()

        # Token indices:
        # 0: Q:1, 1: Q:2, 2: Q:UNK, 3: STAGE:taken, 4: STAGE:UNK, 5: MEDS_BIRTH, 6: glucose
        assert 5 in tokens  # MEDS_BIRTH
        assert 6 in tokens  # glucose (appears twice)
        assert 0 in tokens  # Q:1 (for 90.0)
        assert 1 in tokens  # Q:2 (for 100.0)
        assert 3 in tokens  # STAGE:taken
        assert 4 in tokens  # STAGE:UNK (for unknown_stage)


class TestFactorizedModeWithText:
    """Test factorized mode with text tokens."""

    @pytest.fixture
    def meds_dataset_with_text(self, tmp_path, convert_meds_to_reader):
        """Create MEDS Reader dataset with text_value field."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        birth_time = datetime(2000, 1, 1)
        records = [
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
                "code": "DRUG/metformin",
                "numeric_value": None,
                "text_value": "oral",
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=60),
                "code": "DRUG/insulin",
                "numeric_value": None,
                "text_value": "subcutaneous injection",
                "description": "",
                "unit": None,
            },
        ]

        df = pl.DataFrame(records).with_columns([pl.col("unit").cast(pl.Int32)])
        df.write_parquet(data_dir / "data_0.parquet")

        codes = pl.DataFrame(
            {"code": ["MEDS_BIRTH", "DRUG/metformin", "DRUG/insulin"], "description": ["", "", ""]}
        )
        codes.write_parquet(metadata_dir / "codes.parquet")

        dataset_json = {"dataset_name": "test", "dataset_version": "1.0"}
        with open(metadata_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        samples = pl.DataFrame({"id": [1], "index_t": [birth_time + timedelta(days=100)], "split": ["train"]})
        samples.write_parquet(metadata_dir / "samples.parquet")

        reader_path = convert_meds_to_reader(tmp_path)
        return reader_path

    def test_text_tokens_emitted(self, meds_dataset_with_text, tmp_path):
        """Test that TXT:* tokens are emitted for text_value."""
        vocab_entries = [
            {"type": "text", "text_string": "oral", "weight": -1.0},
            {"type": "text", "text_string": "subcutaneous_injection", "weight": -1.0},
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "metformin", "weight": -0.4},
            {"type": "code", "code_string": "insulin", "weight": -0.3},
        ]
        vocab_data = {
            "vocab": vocab_entries,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
            "config": {
                "tokenization_mode": "factorized",
                "emit_quantiles": False,
                "emit_text": True,
                "emit_stage": False,
                "remove_prefixes": True,
                "separator": "/",
            },
            "quantile_breaks": {},
            "discovered_stages": [],
        }
        vocab_path = tmp_path / "vocab_with_text.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)

        out_dir = tmp_path / "pretokenized"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=meds_dataset_with_text,
            samples_path=meds_dataset_with_text / "metadata" / "samples.parquet",
            split="train",
            num_workers=1,
        )

        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        tokens = df["token_ids"][0].to_list()

        # Token indices:
        # 0: TXT:oral, 1: TXT:subcutaneous_injection, 2: MEDS_BIRTH, 3: metformin, 4: insulin
        assert 2 in tokens  # MEDS_BIRTH
        assert 3 in tokens  # metformin
        assert 4 in tokens  # insulin
        assert 0 in tokens  # TXT:oral
        assert 1 in tokens  # TXT:subcutaneous_injection
