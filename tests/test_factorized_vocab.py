"""Tests for FactorizedVocab class."""

from datetime import datetime, timedelta

import pytest

from ehr_fm.vocabulary import FactorizedVocab


class TestFactorizedVocab:
    """Test FactorizedVocab training and output."""

    @pytest.fixture
    def synthetic_events(self):
        """Create synthetic patient events for vocabulary training."""
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

        # Patient 2: More glucose values
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

    def test_creates_quantile_tokens(self, synthetic_events):
        """FactorizedVocab creates Q:1...Q:n tokens."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, num_quantiles=5, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab_entries, _ = vocab.get_vocab()

        # Check quantile tokens present
        quantile_labels = [e["label"] for e in vocab_entries if e["type"] == "quantile"]
        assert "Q:1" in quantile_labels
        assert "Q:5" in quantile_labels
        assert "Q:UNK" in quantile_labels
        assert len(quantile_labels) == 6  # Q:1 through Q:5 + Q:UNK

    def test_extracts_base_codes(self, synthetic_events):
        """FactorizedVocab extracts base codes with prefix removal."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, remove_prefixes=True, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab_entries, _ = vocab.get_vocab()

        # Check base codes (prefix removed)
        code_strings = [e["code_string"] for e in vocab_entries if e["type"] == "code"]
        assert "glucose" in code_strings  # LAB/glucose → glucose
        assert "diabetes" in code_strings  # DIAGNOSIS/diabetes → diabetes
        assert "metformin" in code_strings  # DRUG/metformin → metformin
        assert "MEDS_BIRTH" in code_strings  # No prefix to remove

    def test_creates_text_tokens(self, synthetic_events):
        """FactorizedVocab creates TXT:* tokens for text values."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, emit_text=True, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab_entries, _ = vocab.get_vocab()

        # Check text tokens
        text_labels = [e["label"] for e in vocab_entries if e["type"] == "text"]
        assert "TXT:oral" in text_labels
        assert "TXT:stable" in text_labels

    def test_computes_quantile_breaks(self, synthetic_events):
        """FactorizedVocab computes quantile breaks for numeric codes."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, num_quantiles=3, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        breaks = vocab.get_quantile_breaks()

        # Should have breaks for LAB/glucose
        assert "LAB/glucose" in breaks
        # With 3 quantiles, should have 2 break points
        assert len(breaks["LAB/glucose"]) == 2

    def test_save_includes_quantile_breaks(self, synthetic_events, tmp_path):
        """Saved vocab.json includes quantile_breaks."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, num_quantiles=5, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab.save(tmp_path, overwrite=True)

        # Load and verify
        import json

        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "quantile_breaks" in data
        assert "LAB/glucose" in data["quantile_breaks"]
        assert data["config"]["tokenization_mode"] == "factorized"
        assert data["config"]["num_quantiles"] == 5

    def test_remove_prefixes_false(self, synthetic_events):
        """With remove_prefixes=False, codes kept as-is."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, remove_prefixes=False, emit_stage=False)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab_entries, _ = vocab.get_vocab()

        # Check full codes preserved
        code_strings = [e["code_string"] for e in vocab_entries if e["type"] == "code"]
        assert "LAB/glucose" in code_strings
        assert "DIAGNOSIS/diabetes" in code_strings

    def test_vocab_order(self, synthetic_events):
        """Vocab entries in expected order: quantiles, stages, text, codes."""
        vocab_config = {
            "vocab_size": 100,
            "n_numeric_bins": 5,
            "numeric_reservoir_size": 100,
            "seed": 42,
            "n_samples": len(synthetic_events),
        }

        vocab = FactorizedVocab(vocab_config, num_quantiles=3, emit_stage=True)

        for patient_events in synthetic_events:
            vocab.forward([patient_events])

        vocab_entries, _ = vocab.get_vocab()

        # Find first index of each type
        quantile_idx = next(i for i, e in enumerate(vocab_entries) if e["type"] == "quantile")
        stage_idx = next(i for i, e in enumerate(vocab_entries) if e["type"] == "stage")
        text_idx = next(i for i, e in enumerate(vocab_entries) if e["type"] == "text")
        code_idx = next(i for i, e in enumerate(vocab_entries) if e["type"] == "code")

        # Order: quantile < stage < text < code
        assert quantile_idx < stage_idx < text_idx < code_idx
