"""Utilities for converting and verifying MEDS Reader datasets.

Wraps the ``meds_reader_convert`` and ``meds_reader_verify`` CLI commands
provided by the ``meds-reader`` package.
"""

import shutil
import subprocess
from pathlib import Path

from .logger import setup_logging

logger = setup_logging(child_name="meds_reader_utils")


def check_meds_reader_commands() -> bool:
    """Return True if both ``meds_reader_convert`` and ``meds_reader_verify`` are on PATH."""
    for cmd in ("meds_reader_convert", "meds_reader_verify"):
        try:
            subprocess.run([cmd, "--help"], capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"Command {cmd} is not available or failed: {e}")
            return False
    return True


def convert_to_meds_reader(source_dir: Path, target_dir: Path) -> None:
    """Convert a MEDS dataset directory to MEDS Reader format.

    Args:
        source_dir: Path to a MEDS dataset (must contain ``data/`` with parquet files).
        target_dir: Destination for the MEDS Reader database.

    Raises:
        RuntimeError: If ``meds_reader_convert`` is not available or fails.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not check_meds_reader_commands():
        raise RuntimeError(
            "meds_reader_convert / meds_reader_verify are not available. "
            "Ensure the meds-reader package is installed."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["meds_reader_convert", str(source_dir), str(target_dir)]
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.info(f"meds_reader_convert stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"meds_reader_convert stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"meds_reader_convert failed (rc={e.returncode}):\n" f"stdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e

    metadata_src = source_dir / "metadata"
    if metadata_src.is_dir():
        shutil.copytree(metadata_src, target_dir / "metadata", dirs_exist_ok=True)

    logger.info(f"Converted MEDS to MEDS Reader: {target_dir}")


def verify_meds_reader(meds_dir: Path, reader_dir: Path) -> None:
    """Verify a MEDS Reader dataset against its source MEDS dataset.

    Args:
        meds_dir: Path to the original MEDS dataset.
        reader_dir: Path to the MEDS Reader database to verify.

    Raises:
        RuntimeError: If ``meds_reader_verify`` fails.
    """
    meds_dir = Path(meds_dir)
    reader_dir = Path(reader_dir)

    cmd = ["meds_reader_verify", str(meds_dir), str(reader_dir)]
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.info(f"meds_reader_verify stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"meds_reader_verify stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"meds_reader_verify failed (rc={e.returncode}):\n" f"stdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e

    logger.info("MEDS Reader verification passed")
