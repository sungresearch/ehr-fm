import subprocess
from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest

from ehr_fm.meds_reader_utils import (
    check_meds_reader_commands,
    convert_to_meds_reader,
    verify_meds_reader,
)


def _create_meds_dir(base_dir):
    """Create a minimal MEDS dataset at base_dir with data/ and metadata/."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True)
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir()

    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "code": ["MEDS_BIRTH", "LAB/glucose"],
            "numeric_value": [None, 5.0],
            "text_value": ["birth", "lab_result"],
            "description": ["Birth event", "Glucose lab"],
        }
    )
    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))

    df.write_parquet(data_dir / "0.parquet")

    samples = pl.DataFrame({"id": [1], "index_t": [datetime(2024, 1, 2)]})
    samples.write_parquet(metadata_dir / "samples.parquet")
    return base_dir


class TestCheckMedsReaderCommands:
    def test_commands_available(self):
        assert check_meds_reader_commands() is True

    def test_returns_false_when_command_missing(self):
        """A binary not on PATH (FileNotFoundError) is reported as unavailable."""
        with patch(
            "ehr_fm.meds_reader_utils.subprocess.run",
            side_effect=FileNotFoundError("meds_reader_convert"),
        ):
            assert check_meds_reader_commands() is False

    def test_returns_false_on_nonzero_exit(self):
        """A command that exists but exits non-zero is also treated as unavailable."""
        err = subprocess.CalledProcessError(returncode=1, cmd="meds_reader_convert")
        with patch("ehr_fm.meds_reader_utils.subprocess.run", side_effect=err):
            assert check_meds_reader_commands() is False


class TestConvertToMedsReader:
    def test_valid_conversion(self, tmp_path):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_dir = tmp_path / "reader"
        convert_to_meds_reader(meds_dir, reader_dir)
        assert reader_dir.is_dir()

    def test_metadata_copied(self, tmp_path):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_dir = tmp_path / "reader"
        convert_to_meds_reader(meds_dir, reader_dir)
        assert (reader_dir / "metadata" / "samples.parquet").exists()

    def test_invalid_source_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "data").mkdir()

        with pytest.raises(RuntimeError):
            convert_to_meds_reader(empty_dir, tmp_path / "reader_fail")


class TestVerifyMedsReader:
    def test_valid_verification(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_dir = convert_meds_to_reader(meds_dir)
        verify_meds_reader(meds_dir, reader_dir)
