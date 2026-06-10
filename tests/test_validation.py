"""Tests for ehr_fm.validation -- PathValidator, validate_config, and config models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ehr_fm.data import MEDSReaderDatasetConfig
from ehr_fm.validation import PathValidator, VocabConfig, validate_config


class TestPathValidator:
    def test_existing_file_ok(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("x")
        assert PathValidator(path=f, ptype="file").path == f

    def test_existing_dir_ok(self, tmp_path):
        assert PathValidator(path=tmp_path, ptype="dir").path == tmp_path

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PathValidator(path=tmp_path / "nope.parquet", ptype="file")

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PathValidator(path=tmp_path / "nope", ptype="dir")

    def test_extension_match_ok(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("x")
        assert PathValidator(path=f, ptype="file", extensions=".parquet").path == f

    def test_extension_mismatch_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x")
        with pytest.raises(ValidationError):
            PathValidator(path=f, ptype="file", extensions=".parquet")

    def test_extra_field_forbidden(self, tmp_path):
        with pytest.raises(ValidationError):
            PathValidator(path=tmp_path, ptype="dir", bogus=1)


class TestValidateConfig:
    def test_dict_input_constructs_model(self):
        cfg = validate_config({"n_samples": 3, "vocab_size": 7}, VocabConfig)
        assert isinstance(cfg, VocabConfig)
        assert cfg.n_samples == 3
        assert cfg.vocab_size == 7

    def test_model_instance_passthrough(self):
        original = VocabConfig(n_samples=1, vocab_size=2)
        assert validate_config(original, VocabConfig) is original

    def test_wrong_model_type_raises_type_error(self):
        other = VocabConfig(n_samples=1, vocab_size=2)
        with pytest.raises(TypeError):
            validate_config(other, MEDSReaderDatasetConfig)

    def test_invalid_dict_raises(self):
        with pytest.raises(ValidationError):
            validate_config({}, VocabConfig)  # missing required fields

    def test_empty_config_file_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValidationError):
            validate_config(p, VocabConfig)


class TestMEDSReaderDatasetConfig:
    def test_missing_reader_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MEDSReaderDatasetConfig(meds_reader_path=tmp_path / "nope")

    def test_defaults_samples_path_under_metadata(self, tmp_path):
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "samples.parquet").write_text("x")
        cfg = MEDSReaderDatasetConfig(meds_reader_path=str(tmp_path))
        assert isinstance(cfg.meds_reader_path, Path)
        assert cfg.samples_path == tmp_path / "metadata" / "samples.parquet"


class TestVocabConfig:
    def test_defaults(self):
        cfg = VocabConfig(n_samples=5, vocab_size=10)
        assert cfg.n_numeric_bins == 10
        assert cfg.numeric_reservoir_size == 10000
        assert cfg.numeric_bin_by_unit is False
        assert cfg.seed is None

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            VocabConfig(n_samples=5)  # missing vocab_size
