"""Core transform pipeline for MEDS datasets.

Reads a MEDS dataset (``data/`` + ``metadata/`` directory structure), applies
registered transforms to the parquet files, copies source metadata unchanged,
and optionally converts the result to MEDS Reader format.

Currently the only supported transform is :class:`~.visit_time.VisitEndTimeMover`,
which fixes data leakage in OMOP-derived datasets (e.g. MIMIC) where diagnosis,
procedure, and observation events are incorrectly timestamped at visit start
instead of visit end.
"""

import shutil
from pathlib import Path
from typing import Any

import polars as pl

from ..logger import setup_logging
from ..meds_reader_utils import convert_to_meds_reader, verify_meds_reader
from .validation import TransformConfig, validate_transform_config

logger = setup_logging(child_name="transforms.core")

_REQUIRED_MEDS_COLUMNS = {"subject_id", "time", "code"}


def _has_meds_columns(df: pl.DataFrame) -> bool:
    """Check that *df* contains the required MEDS columns."""
    return _REQUIRED_MEDS_COLUMNS.issubset(set(df.columns))


class MEDSTransformPipeline:
    """Apply transforms to a MEDS dataset and write a new MEDS dataset.

    The pipeline expects a standard MEDS directory layout as input::

        input_dir/
        ├── data/          # parquet files with subject_id, time, code, ...
        └── metadata/      # codes.parquet, dataset.json, samples.parquet, ...

    Output mirrors the same structure.  Source ``metadata/`` is copied verbatim
    because the current transforms only modify timestamps, not codes or schema.
    """

    def __init__(
        self,
        config: dict[str, Any] | TransformConfig,
        dataset_name: str = "Transformed MEDS Dataset",
        dataset_version: str = "version",
    ):
        if isinstance(config, dict):
            self.config = validate_transform_config(config)
        else:
            self.config = config

        self.dataset_name = dataset_name
        self.dataset_version = dataset_version

        self.convert_to_meds_reader_flag = self.config.convert_to_meds_reader
        self.verify_meds_reader_flag = self.config.verify_meds_reader
        self.meds_reader_output_dir = self.config.meds_reader_output_dir

        self.transforms: list = []
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Initialize transform stages based on config."""
        if self.config.move_to_visit_end:
            from .visit_time import VisitEndTimeMover

            self.transforms.append(VisitEndTimeMover(self.config.visit_end_time_config))

    @staticmethod
    def _create_meds_structure(output_dir: Path) -> tuple[Path, Path]:
        """Create ``data/`` and ``metadata/`` directories under *output_dir*."""
        data_dir = output_dir / "data"
        metadata_dir = output_dir / "metadata"
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created MEDS structure: {data_dir} and {metadata_dir}")
        return data_dir, metadata_dir

    @staticmethod
    def _copy_metadata(input_dir: Path, metadata_dir: Path) -> None:
        """Copy source ``metadata/`` to the output directory.

        Looks for ``metadata/`` under *input_dir*; if absent, logs a warning.
        """
        source_metadata = input_dir / "metadata"
        if not source_metadata.is_dir():
            logger.warning(f"Source metadata directory not found: {source_metadata}")
            return

        shutil.copytree(source_metadata, metadata_dir, dirs_exist_ok=True)
        logger.info(f"Copied metadata from {source_metadata} to {metadata_dir}")

    def transform_parquet_files(
        self, input_dir: Path, output_dir: Path, file_pattern: str = "*.parquet"
    ) -> None:
        """Transform all parquet files in a MEDS dataset.

        Args:
            input_dir: Root of the source MEDS dataset.
            output_dir: Root for the transformed MEDS dataset.
            file_pattern: Glob pattern for parquet files inside ``data/``.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir, metadata_dir = self._create_meds_structure(output_dir)

        parquet_files = list(input_dir.glob(file_pattern))
        if not parquet_files:
            parquet_files = list(input_dir.glob("data/*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {input_dir} with pattern {file_pattern}")

        logger.info(f"Processing {len(parquet_files)} files from {input_dir} to {data_dir}")

        for file_path in parquet_files:
            logger.info(f"Processing {file_path.name}...")

            df = pl.read_parquet(file_path)

            if not _has_meds_columns(df):
                logger.warning(f"{file_path.name} does not have valid MEDS schema")
                continue

            original_rows = len(df)

            for i, transform in enumerate(self.transforms):
                logger.info(f"  Applying transform {i+1}/{len(self.transforms)}: {type(transform).__name__}")
                df = transform(df)

            final_rows = len(df)
            if final_rows != original_rows:
                logger.info(f"  Row count changed: {original_rows} \u2192 {final_rows}")

            output_path = data_dir / file_path.name
            df = df.sort(["subject_id", "time"], maintain_order=True)
            df.write_parquet(output_path, use_pyarrow=True)
            logger.info(f"  Saved to {output_path}")

        self._copy_metadata(input_dir, metadata_dir)

        if self.convert_to_meds_reader_flag:
            reader_dir = self.meds_reader_output_dir
            if not reader_dir:
                reader_dir = output_dir.parent / f"{output_dir.name}_meds_reader"
            reader_dir = Path(reader_dir)

            convert_to_meds_reader(output_dir, reader_dir)

            if self.verify_meds_reader_flag:
                verify_meds_reader(output_dir, reader_dir)

        logger.info("Transformation pipeline completed!")
