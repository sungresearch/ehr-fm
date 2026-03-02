from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ehr_fm.meds_reader_utils import convert_to_meds_reader


@pytest.fixture
def meds_dataset_dir(tmp_path):
    """Create a minimal MEDS dataset directory with data/ and metadata/.

    Reusable across transforms and meds_reader_utils tests.
    """
    base = tmp_path / "meds_dataset"
    data_dir = base / "data"
    data_dir.mkdir(parents=True)
    metadata_dir = base / "metadata"
    metadata_dir.mkdir()

    birth_time = datetime(2000, 1, 1)
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [birth_time, birth_time + timedelta(days=30), birth_time + timedelta(days=60)],
            "code": ["MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/flu"],
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
            "index_t": [birth_time + timedelta(days=90)],
            "split": ["train"],
        }
    )
    samples.write_parquet(metadata_dir / "samples.parquet")
    return base


@pytest.fixture
def convert_meds_to_reader():
    """Return a helper that converts a MEDS dataset directory to MEDS Reader format.

    Casts ``pl.Null`` columns to ``pl.String`` so ``meds_reader_convert`` succeeds,
    then delegates to :func:`ehr_fm.meds_reader_utils.convert_to_meds_reader`
    (which also copies ``metadata/`` into the resulting database).
    """

    def _convert(meds_path: Path) -> Path:
        _fixup_null_columns(meds_path / "data")

        reader_path = meds_path / "meds_reader_db"
        convert_to_meds_reader(meds_path, reader_path)
        return reader_path

    return _convert


def _fixup_null_columns(data_dir: Path) -> None:
    """Re-write parquet files, casting ``pl.Null`` columns to ``pl.String``."""
    for pq in data_dir.glob("*.parquet"):
        df = pl.read_parquet(pq)
        casts = {col: pl.String for col, dtype in df.schema.items() if dtype == pl.Null}
        if casts:
            df = df.with_columns([pl.col(c).cast(t) for c, t in casts.items()])
            df.write_parquet(pq)
