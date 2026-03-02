from datetime import datetime

import polars as pl
import pytest

from ehr_fm.transforms.core import MEDSTransformPipeline, _has_meds_columns
from ehr_fm.transforms.validation import TransformConfig


def _write_meds_dataset(base_dir, *, include_visit_data: bool = False):
    """Write a minimal MEDS dataset under base_dir/data/ and base_dir/metadata/."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    visit_start = datetime(2024, 1, 1, 8, 0)
    visit_end = datetime(2024, 1, 3, 16, 0)

    rows = {
        "subject_id": [1, 1, 1],
        "time": [visit_start, visit_start, visit_start],
        "code": ["MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/flu"],
        "numeric_value": [None, 5.0, None],
        "text_value": ["", "", ""],
        "description": ["", "", ""],
    }
    if include_visit_data:
        rows["subject_id"] = [1, 1, 1, 1, 1]
        rows["time"] = [visit_start, visit_end, visit_start, visit_start, visit_start]
        rows["code"] = ["VISIT/IP", "VISIT/IP", "MEDS_BIRTH", "LAB/glucose", "DIAGNOSIS/flu"]
        rows["numeric_value"] = [None, None, None, 5.0, None]
        rows["text_value"] = ["", "", "", "", ""]
        rows["description"] = ["", "", "", "", ""]
        rows["visit_id"] = [100, 100, None, 100, 100]
        rows["workflow_stage"] = ["start", "end", None, None, None]

    df = pl.DataFrame(rows)
    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))
    df.write_parquet(data_dir / "0.parquet")

    samples = pl.DataFrame({"id": [1], "index_t": [visit_start]})
    samples.write_parquet(metadata_dir / "samples.parquet")
    return base_dir


class TestHasMedsColumns:
    def test_valid_columns(self):
        df = pl.DataFrame({"subject_id": [1], "time": [datetime.now()], "code": ["X"]})
        assert _has_meds_columns(df) is True

    def test_missing_column(self):
        df = pl.DataFrame({"subject_id": [1], "time": [datetime.now()]})
        assert _has_meds_columns(df) is False

    def test_extra_columns_ok(self):
        df = pl.DataFrame({"subject_id": [1], "time": [datetime.now()], "code": ["X"], "extra": [1]})
        assert _has_meds_columns(df) is True


class TestMEDSTransformPipelineInit:
    def test_dict_config(self):
        pipeline = MEDSTransformPipeline(config={})
        assert isinstance(pipeline.config, TransformConfig)
        assert len(pipeline.transforms) == 0

    def test_transform_config_object(self):
        cfg = TransformConfig(move_to_visit_end=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        assert pipeline.config is cfg

    def test_move_to_visit_end_adds_transform(self):
        cfg = TransformConfig(move_to_visit_end=True)
        pipeline = MEDSTransformPipeline(config=cfg)
        assert len(pipeline.transforms) == 1

    def test_no_transforms_when_disabled(self):
        cfg = TransformConfig(move_to_visit_end=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        assert len(pipeline.transforms) == 0


class TestCreateMedsStructure:
    def test_creates_data_and_metadata_dirs(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        data_dir, metadata_dir = MEDSTransformPipeline._create_meds_structure(out)
        assert data_dir.is_dir()
        assert metadata_dir.is_dir()
        assert data_dir == out / "data"
        assert metadata_dir == out / "metadata"


class TestCopyMetadata:
    def test_copies_metadata_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "metadata").mkdir()
        (src / "metadata" / "codes.parquet").touch()

        dst_metadata = tmp_path / "dst_metadata"
        dst_metadata.mkdir()

        MEDSTransformPipeline._copy_metadata(src, dst_metadata)
        assert (dst_metadata / "codes.parquet").exists()

    def test_missing_metadata_warns(self, tmp_path):
        src = tmp_path / "src_no_meta"
        src.mkdir()
        dst_metadata = tmp_path / "dst_metadata"
        dst_metadata.mkdir()
        MEDSTransformPipeline._copy_metadata(src, dst_metadata)
        assert not (dst_metadata / "codes.parquet").exists()


class TestTransformParquetFiles:
    def test_no_parquet_files_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pipeline = MEDSTransformPipeline(config={})
        with pytest.raises(ValueError, match="No parquet files found"):
            pipeline.transform_parquet_files(empty_dir, tmp_path / "out")

    def test_passthrough_no_transforms(self, tmp_path):
        """With no transforms, data is written unchanged (except sort)."""
        src = _write_meds_dataset(tmp_path / "src")
        out = tmp_path / "out"

        cfg = TransformConfig(convert_to_meds_reader=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, out)

        result = pl.read_parquet(out / "data" / "0.parquet")
        assert len(result) == 3
        assert set(result.columns) >= {"subject_id", "time", "code"}

    def test_metadata_copied(self, tmp_path):
        src = _write_meds_dataset(tmp_path / "src")
        out = tmp_path / "out"

        cfg = TransformConfig(convert_to_meds_reader=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, out)

        assert (out / "metadata" / "samples.parquet").exists()

    def test_invalid_schema_skipped(self, tmp_path):
        """A parquet file without MEDS columns is skipped."""
        src = tmp_path / "src"
        data_dir = src / "data"
        data_dir.mkdir(parents=True)
        (src / "metadata").mkdir()

        bad_df = pl.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        bad_df.write_parquet(data_dir / "bad.parquet")

        cfg = TransformConfig(convert_to_meds_reader=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, tmp_path / "out")

        assert not (tmp_path / "out" / "data" / "bad.parquet").exists()

    def test_visit_end_transform_integration(self, tmp_path):
        """With move_to_visit_end, diagnosis events are moved to visit end time."""
        src = _write_meds_dataset(tmp_path / "src", include_visit_data=True)
        out = tmp_path / "out"

        cfg = TransformConfig(move_to_visit_end=True, convert_to_meds_reader=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, out)

        result = pl.read_parquet(out / "data" / "0.parquet")
        diag_row = result.filter(pl.col("code") == "DIAGNOSIS/flu")
        visit_end = datetime(2024, 1, 3, 16, 0)
        assert diag_row["time"][0] == visit_end

    def test_meds_reader_conversion(self, tmp_path):
        """With convert_to_meds_reader=True, a reader directory is created."""
        src = _write_meds_dataset(tmp_path / "src")
        out = tmp_path / "out"

        cfg = TransformConfig(convert_to_meds_reader=True, verify_meds_reader=False)
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, out)

        expected_reader = out.parent / f"{out.name}_meds_reader"
        assert expected_reader.is_dir()

    def test_custom_meds_reader_output_dir(self, tmp_path):
        src = _write_meds_dataset(tmp_path / "src")
        out = tmp_path / "out"
        custom_reader = tmp_path / "my_reader"

        cfg = TransformConfig(
            convert_to_meds_reader=True,
            verify_meds_reader=False,
            meds_reader_output_dir=custom_reader,
        )
        pipeline = MEDSTransformPipeline(config=cfg)
        pipeline.transform_parquet_files(src, out)

        assert custom_reader.is_dir()
