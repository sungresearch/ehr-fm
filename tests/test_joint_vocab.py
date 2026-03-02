"""Tests for JointVocab class."""

from datetime import datetime, timedelta

import pytest

from ehr_fm.vocabulary import JointVocab


class TestJointVocabInit:
    """Test JointVocab initialization."""

    @pytest.fixture
    def base_config(self):
        """Minimal vocab config."""
        return {"vocab_size": 100, "n_samples": 10}

    @pytest.fixture
    def sample_breaks(self):
        """Sample quantile breaks."""
        return {"LAB/glucose": [70.0, 100.0, 130.0]}

    def test_init_with_breaks(self, base_config, sample_breaks):
        """Initialize with quantile breaks."""
        vocab = JointVocab(
            config=base_config,
            quantile_breaks=sample_breaks,
        )
        assert vocab.quantile_breaks == sample_breaks
        assert vocab.vocab_size == 100

    def test_init_with_discovered_stages(self, base_config, sample_breaks):
        """Initialize with discovered stages."""
        vocab = JointVocab(
            config=base_config,
            quantile_breaks=sample_breaks,
            discovered_stages=["taken", "order"],
        )
        assert vocab.discovered_stages == {"taken", "order"}

    def test_init_with_emit_flags(self, base_config, sample_breaks):
        """Initialize with custom emit flags."""
        vocab = JointVocab(
            config=base_config,
            quantile_breaks=sample_breaks,
            emit_quantiles=False,
            emit_text=False,
            emit_stage=True,
        )
        assert vocab.emit_quantiles is False
        assert vocab.emit_text is False
        assert vocab.emit_stage is True


class TestJointVocabForward:
    """Test JointVocab.forward() batch processing."""

    @pytest.fixture
    def vocab(self):
        """JointVocab with all emissions enabled."""
        return JointVocab(
            config={"vocab_size": 100, "n_samples": 2},
            quantile_breaks={"LAB/glucose": [100.0]},
            discovered_stages=["taken"],
            emit_quantiles=True,
            emit_text=True,
            emit_stage=True,
            remove_prefixes=True,
        )

    def test_forward_accumulates_weights(self, vocab):
        """Forward accumulates token weights."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
            ],
        ]
        vocab.forward(batch)

        # Should have accumulated weights for combined tokens
        assert "MEDS_BIRTH" in vocab.combined_weights
        assert "glucose/Q:1" in vocab.combined_weights

    def test_forward_with_text(self, vocab):
        """Forward handles text values."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "DRUG/metformin", "time": birth_time + timedelta(days=30), "text_value": "oral"},
            ],
        ]
        vocab.forward(batch)

        assert "metformin/TXT:oral" in vocab.combined_weights

    def test_forward_with_stage(self, vocab):
        """Forward handles workflow_stage."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "workflow_stage": "taken"},
            ],
        ]
        vocab.forward(batch)

        assert "glucose/STAGE:taken" in vocab.combined_weights

    def test_forward_updates_age_stats(self, vocab):
        """Forward updates age statistics."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": None},
            ],
        ]
        vocab.forward(batch)

        # Age stats should have been updated (30 days in seconds)
        assert vocab.age_stats.count > 0
        assert vocab.age_stats.mean > 0

    def test_forward_empty_batch(self, vocab):
        """Forward handles empty batch gracefully."""
        vocab.forward([[]])
        # Should not crash, just log warning
        assert len(vocab.combined_weights) == 0


class TestJointVocabGetVocab:
    """Test JointVocab.get_vocab() output generation."""

    @pytest.fixture
    def trained_vocab(self):
        """JointVocab with accumulated weights."""
        vocab = JointVocab(
            config={"vocab_size": 10, "n_samples": 1},
            quantile_breaks={"LAB/glucose": [100.0]},
            emit_quantiles=True,
            emit_text=False,
            emit_stage=False,
            remove_prefixes=True,
        )
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=60), "numeric_value": 120.0},
            ],
        ]
        vocab.forward(batch)
        return vocab

    def test_get_vocab_returns_tuple(self, trained_vocab):
        """get_vocab returns tuple of (entries, num_event_codes)."""
        result = trained_vocab.get_vocab()
        assert isinstance(result, tuple)
        assert len(result) == 2
        entries, num_event_codes = result
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert isinstance(num_event_codes, int)

    def test_vocab_entries_have_required_fields(self, trained_vocab):
        """Each entry has type, code_string, weight."""
        entries, _ = trained_vocab.get_vocab()
        for entry in entries:
            assert "type" in entry
            assert "code_string" in entry
            assert "weight" in entry
            assert entry["type"] == "code"

    def test_vocab_sorted_by_weight(self, trained_vocab):
        """Entries sorted by weight (ascending = most informative first)."""
        entries, _ = trained_vocab.get_vocab()
        weights = [e["weight"] for e in entries]
        assert weights == sorted(weights)

    def test_vocab_contains_combined_tokens(self, trained_vocab):
        """Vocab contains combined tokens like glucose/Q:1."""
        entries, _ = trained_vocab.get_vocab()
        code_strings = [e["code_string"] for e in entries]
        assert "glucose/Q:1" in code_strings
        assert "glucose/Q:2" in code_strings
        assert "MEDS_BIRTH" in code_strings

    def test_vocab_truncated_to_size(self):
        """Vocab truncated to vocab_size."""
        vocab = JointVocab(
            config={"vocab_size": 2, "n_samples": 1},
            quantile_breaks={"LAB/glucose": [100.0]},
        )
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=60), "numeric_value": 120.0},
            ],
        ]
        vocab.forward(batch)
        entries, _ = vocab.get_vocab()
        assert len(entries) <= 2


class TestJointVocabSave:
    """Test JointVocab.save() file output."""

    @pytest.fixture
    def trained_vocab(self):
        """JointVocab with accumulated weights."""
        vocab = JointVocab(
            config={"vocab_size": 10, "n_samples": 1},
            quantile_breaks={"LAB/glucose": [100.0]},
            discovered_stages=["taken"],
            emit_quantiles=True,
            emit_text=True,
            emit_stage=True,
            remove_prefixes=True,
            separator="/",
        )
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
            ],
        ]
        vocab.forward(batch)
        return vocab

    def test_save_creates_file(self, trained_vocab, tmp_path):
        """save creates vocab.json file."""
        trained_vocab.save(tmp_path)
        assert (tmp_path / "vocab.json").exists()

    def test_save_contains_vocab_entries(self, trained_vocab, tmp_path):
        """Saved file contains vocab entries."""
        import json

        trained_vocab.save(tmp_path)
        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "vocab" in data
        assert len(data["vocab"]) > 0

    def test_save_contains_age_stats(self, trained_vocab, tmp_path):
        """Saved file contains age statistics."""
        import json

        trained_vocab.save(tmp_path)
        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "age_stats" in data
        assert "mean" in data["age_stats"]
        assert "std" in data["age_stats"]

    def test_save_contains_config(self, trained_vocab, tmp_path):
        """Saved file contains config section."""
        import json

        trained_vocab.save(tmp_path)
        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "config" in data
        assert data["config"]["tokenization_mode"] == "joint"
        assert data["config"]["emit_quantiles"] is True
        assert data["config"]["emit_text"] is True
        assert data["config"]["emit_stage"] is True
        assert data["config"]["remove_prefixes"] is True
        assert data["config"]["separator"] == "/"

    def test_save_contains_quantile_breaks(self, trained_vocab, tmp_path):
        """Saved file contains quantile breaks."""
        import json

        trained_vocab.save(tmp_path)
        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "quantile_breaks" in data
        assert "LAB/glucose" in data["quantile_breaks"]

    def test_save_contains_discovered_stages(self, trained_vocab, tmp_path):
        """Saved file contains discovered stages."""
        import json

        trained_vocab.save(tmp_path)
        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        assert "discovered_stages" in data
        assert "taken" in data["discovered_stages"]


class TestJointVocabIntegration:
    """Integration tests for JointVocab with full workflow."""

    def test_two_pass_training_workflow(self, tmp_path):
        """Full two-pass training: QuantilePreScanner → JointVocab."""
        from ehr_fm.vocabulary import QuantilePreScanner

        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=30),
                    "numeric_value": 80.0,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=60),
                    "numeric_value": 120.0,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=90),
                    "numeric_value": 180.0,
                    "workflow_stage": "order",
                },
            ],
        ]

        # Pass 1: Pre-scan to compute breaks and discover stages
        prescanner = QuantilePreScanner(num_quantiles=3)
        prescanner.forward(batch)
        quantile_breaks = prescanner.compute_breaks()
        discovered_stages = prescanner.get_discovered_stages()

        # Pass 2: Train JointVocab with pre-computed breaks
        vocab = JointVocab(
            config={"vocab_size": 50, "n_samples": 1},
            quantile_breaks=quantile_breaks,
            discovered_stages=discovered_stages,
            num_quantiles=3,
            emit_quantiles=True,
            emit_stage=True,
            emit_text=False,
        )
        vocab.forward(batch)

        # Save and verify
        vocab.save(tmp_path)

        import json

        with open(tmp_path / "vocab.json") as f:
            data = json.load(f)

        # Should have combined tokens with quantiles and stages
        code_strings = [e["code_string"] for e in data["vocab"]]
        assert "MEDS_BIRTH" in code_strings

        # Should have glucose tokens with quantile and stage combinations
        glucose_tokens = [s for s in code_strings if s.startswith("glucose")]
        assert len(glucose_tokens) > 0

        # Check metadata
        assert "LAB/glucose" in data["quantile_breaks"]
        assert set(data["discovered_stages"]) == {"order", "taken"}
