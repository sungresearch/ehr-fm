"""Tests for factorized vocabulary helper functions."""

from ehr_fm.defaults import STAGE_UNK_LABEL, quantile_labels
from ehr_fm.vocabulary import (
    _factorized_quantile_vocab_entries,
    _factorized_stage_vocab_entries,
)


class TestQuantileLabels:
    """Test quantile_labels function."""

    def test_default_10_quantiles(self):
        """Default generates Q:1 through Q:10 plus Q:UNK."""
        labels = quantile_labels()
        assert len(labels) == 11
        assert labels[0] == "Q:1"
        assert labels[9] == "Q:10"
        assert labels[10] == "Q:UNK"

    def test_custom_quantile_count(self):
        """Custom count generates correct number of labels."""
        labels = quantile_labels(5)
        assert len(labels) == 6
        assert labels == ["Q:1", "Q:2", "Q:3", "Q:4", "Q:5", "Q:UNK"]

    def test_single_quantile(self):
        """Single quantile generates Q:1 and Q:UNK."""
        labels = quantile_labels(1)
        assert labels == ["Q:1", "Q:UNK"]


class TestFactorizedQuantileVocabEntries:
    """Test _factorized_quantile_vocab_entries function."""

    def test_creates_correct_entries(self):
        """Creates vocab entries with correct structure."""
        entries = _factorized_quantile_vocab_entries(5)
        assert len(entries) == 6

        # Check structure
        for entry in entries:
            assert entry["type"] == "quantile"
            assert "label" in entry
            assert entry["weight"] == -1.0

        # Check labels
        labels = [e["label"] for e in entries]
        assert "Q:1" in labels
        assert "Q:5" in labels
        assert "Q:UNK" in labels


class TestFactorizedStageVocabEntries:
    """Test _factorized_stage_vocab_entries function."""

    def test_creates_entries_from_stages(self):
        """Creates vocab entries from discovered stages."""
        stages = ["order", "taken", "admin"]
        entries = _factorized_stage_vocab_entries(stages)

        # Should have 3 stages + UNK = 4
        assert len(entries) == 4

        labels = [e["label"] for e in entries]
        assert "STAGE:admin" in labels
        assert "STAGE:order" in labels
        assert "STAGE:taken" in labels
        assert STAGE_UNK_LABEL in labels

    def test_normalizes_to_lowercase(self):
        """Stage values normalized to lowercase."""
        stages = ["ORDER", "Taken", "ADMIN"]
        entries = _factorized_stage_vocab_entries(stages)

        labels = [e["label"] for e in entries]
        assert "STAGE:order" in labels
        assert "STAGE:taken" in labels
        assert "STAGE:admin" in labels

    def test_deduplicates_stages(self):
        """Duplicate stages are deduplicated."""
        stages = ["order", "order", "taken"]
        entries = _factorized_stage_vocab_entries(stages)

        labels = [e["label"] for e in entries]
        # Should have 2 unique stages + UNK = 3
        assert len(entries) == 3
        assert labels.count("STAGE:order") == 1

    def test_always_includes_unk(self):
        """Always includes STAGE:UNK."""
        entries = _factorized_stage_vocab_entries([])
        assert len(entries) == 1
        assert entries[0]["label"] == STAGE_UNK_LABEL

    def test_correct_structure(self):
        """Entries have correct structure."""
        entries = _factorized_stage_vocab_entries(["test"])
        for entry in entries:
            assert entry["type"] == "stage"
            assert "label" in entry
            assert entry["weight"] == -1.0
