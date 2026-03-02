"""Extended tests for ehr_fm.data.dataset -- create_dataset factory, _is_meds_reader_format,
MEDSReaderDataset edge cases, and TokenizedDataset modes.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ehr_fm.data.dataset import (
    MEDSReaderDataset,
    TokenizedDataset,
    _is_meds_reader_format,
    create_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_meds_dir(base_dir: Path) -> Path:
    """Create a minimal MEDS dataset with data/ and metadata/."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True)
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir()

    birth_time = datetime(2000, 1, 1)
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [birth_time, birth_time + timedelta(days=10), birth_time + timedelta(days=20)],
            "code": ["MEDS_BIRTH", "LAB/glucose", "DIAG/flu"],
            "numeric_value": [None, 5.0, None],
            "text_value": [None, None, None],
            "description": ["", "", ""],
        }
    )
    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))
    df.write_parquet(data_dir / "0.parquet")

    samples = pl.DataFrame(
        {
            "id": [1],
            "index_t": [birth_time + timedelta(days=30)],
            "split": ["train"],
        }
    )
    samples.write_parquet(metadata_dir / "samples.parquet")
    return base_dir


def _create_tokenized_parquet(path: Path, patients: list[dict]) -> None:
    """Write pretokenized parquet data for TokenizedDataset."""
    rows = []
    for p in patients:
        length = len(p["token_ids"])
        rows.append(
            {
                "subject_id": p.get("subject_id", 1),
                "index_time": p.get("index_time", datetime(2024, 1, 1)),
                "token_ids": pa.array(p["token_ids"], type=pa.int32()),
                "age": pa.array(p.get("age", [float(i) for i in range(length)]), type=pa.float32()),
                "length": length,
                "age_normalized": pa.array(p.get("age_normalized", [0.0] * length), type=pa.float32()),
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


# ---------------------------------------------------------------------------
# _is_meds_reader_format
# ---------------------------------------------------------------------------


class TestIsMedsReaderFormat:
    def test_valid_directory(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_path = convert_meds_to_reader(meds_dir)
        assert _is_meds_reader_format(reader_path) is True

    def test_incomplete_directory(self, tmp_path):
        incomplete = tmp_path / "incomplete_reader"
        incomplete.mkdir()
        (incomplete / "code").mkdir()
        assert _is_meds_reader_format(incomplete) is False

    def test_nonexistent_path(self, tmp_path):
        assert _is_meds_reader_format(tmp_path / "nonexistent") is False


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------


class TestCreateDataset:
    def test_dict_config_valid(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_path = convert_meds_to_reader(meds_dir)
        ds = create_dataset({"dataset_path": str(reader_path)})
        assert isinstance(ds, MEDSReaderDataset)

    def test_dict_config_missing_key_raises(self):
        with pytest.raises(ValueError, match="dataset_path"):
            create_dataset({"some_other_key": "value"})

    def test_non_reader_directory_raises(self, tmp_path):
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        with pytest.raises(ValueError, match="MEDS Reader format"):
            create_dataset({"dataset_path": str(plain_dir)})


# ---------------------------------------------------------------------------
# MEDSReaderDataset
# ---------------------------------------------------------------------------


class TestMEDSReaderDataset:
    def test_getitem_returns_tuple(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_path = convert_meds_to_reader(meds_dir)
        ds = create_dataset({"dataset_path": str(reader_path), "split": "train"})
        item = ds[0]
        assert item is not None
        (sid_tuple, events) = item
        assert isinstance(events, list)
        assert events[0]["code"] == "MEDS_BIRTH"

    def test_split_none_loads_all(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_path = convert_meds_to_reader(meds_dir)
        ds = create_dataset({"dataset_path": str(reader_path)})
        assert len(ds) > 0

    def test_custom_transform(self, tmp_path, convert_meds_to_reader):
        meds_dir = _create_meds_dir(tmp_path / "meds")
        reader_path = convert_meds_to_reader(meds_dir)

        def drop_birth(events):
            return [e for e in events if e["code"] != "MEDS_BIRTH"]

        ds = MEDSReaderDataset(
            {
                "meds_reader_path": str(reader_path),
                "transform": drop_birth,
            }
        )
        _, events = ds[0]
        assert all(e["code"] != "MEDS_BIRTH" for e in events)


# ---------------------------------------------------------------------------
# TokenizedDataset
# ---------------------------------------------------------------------------


class TestTokenizedDatasetOneWindow:
    @pytest.fixture
    def pq_path(self, tmp_path):
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [
                {"subject_id": 1, "token_ids": list(range(10))},
                {"subject_id": 2, "token_ids": list(range(5))},
            ],
        )
        return path

    def test_one_window_per_patient(self, pq_path):
        ds = TokenizedDataset(pq_path, max_length=20, one_window=True)
        assert len(ds) == 2

    def test_max_length_clipping(self, pq_path):
        ds = TokenizedDataset(pq_path, max_length=4, one_window=True)
        item = ds[0]
        assert item["input_ids"].size(0) <= 4


class TestTokenizedDatasetMultiWindow:
    @pytest.fixture
    def pq_path(self, tmp_path):
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [{"subject_id": 1, "token_ids": list(range(20))}],
        )
        return path

    def test_sliding_windows(self, pq_path):
        ds = TokenizedDataset(pq_path, max_length=8, stride=8, one_window=False)
        assert len(ds) > 1

    def test_windows_cover_end(self, pq_path):
        """First window in the index should start at the tail of the sequence."""
        ds = TokenizedDataset(pq_path, max_length=8, stride=8, one_window=False)
        offsets = [ds.window_index[i][1] for i in range(len(ds))]
        assert 0 in offsets  # must include the beginning-of-sequence window


class TestTokenizedDatasetMinLength:
    def test_short_sequences_excluded(self, tmp_path):
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [
                {"subject_id": 1, "token_ids": [0, 1]},  # length 2
                {"subject_id": 2, "token_ids": list(range(10))},  # length 10
            ],
        )
        ds = TokenizedDataset(path, max_length=20, one_window=True, min_length=5)
        assert len(ds) == 1


class TestTokenizedDatasetDropout:
    def test_dropout_reduces_length(self, tmp_path):
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [{"subject_id": 1, "token_ids": list(range(100))}],
        )
        random.seed(42)
        ds = TokenizedDataset(path, max_length=100, one_window=True, dropout_prob=0.5)
        item = ds[0]
        assert item["input_ids"].size(0) < 100

    def test_full_dropout_fallback(self, tmp_path):
        """If dropout removes everything, original slice is used."""
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [{"subject_id": 1, "token_ids": [42]}],
        )
        ds = TokenizedDataset(path, max_length=10, one_window=True, dropout_prob=0.999)
        item = ds[0]
        assert item["input_ids"].size(0) >= 1


class TestTokenizedDatasetPositions:
    def test_convert_ages_to_positions(self, tmp_path):
        path = tmp_path / "tokens.parquet"
        _create_tokenized_parquet(
            path,
            [{"subject_id": 1, "token_ids": [0, 1, 2], "age": [100.0, 200.0, 300.0]}],
        )
        ds = TokenizedDataset(path, max_length=10, one_window=True, convert_ages_to_positions=True)
        item = ds[0]
        assert item["age"].tolist() == [0.0, 1.0, 2.0]
        assert item["age_normalized"].tolist() == [0.0, 0.0, 0.0]
