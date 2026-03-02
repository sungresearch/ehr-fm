#!/usr/bin/env python3
"""
Apply MEDS transformations to parquet files.

The output is a MEDS-compliant dataset with:
- data/ subdirectory containing transformed parquet files
- metadata/ copied verbatim from the source MEDS dataset

Example usage:
    # Apply transformations using config file
    apply_meds_transforms config.yaml

    # Apply visit end time fix (moves events to visit end time)
    apply_meds_transforms --input-dir /path/to/meds --output-dir /path/to/output \
        --move-to-visit-end

    # Apply transformations and skip MEDS Reader conversion
    apply_meds_transforms --input-dir /path/to/meds --output-dir /path/to/output \
        --move-to-visit-end --no-convert-to-meds-reader
"""

import argparse
import sys
from pathlib import Path

from ehr_fm.io import read_json_yaml
from ehr_fm.logger import setup_logging
from ehr_fm.transforms import MEDSTransformPipeline
from ehr_fm.transforms.validation import validate_transform_config


def transform_command(args):
    logger = setup_logging()

    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        logger.info(f"Loading configuration from: {config_path}")
        config = read_json_yaml(config_path)
        if not isinstance(config, dict):
            logger.error("Configuration file must be a YAML/JSON mapping (dictionary)")
            sys.exit(1)

    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    dataset_name = (
        args.dataset_name
        if args.dataset_name is not None
        else config.get("dataset_name", "Transformed Dataset")
    )
    dataset_version = (
        args.dataset_version if args.dataset_version is not None else config.get("dataset_version", "1.0")
    )

    transforms_config = config.get("transforms", {})

    if getattr(args, "move_to_visit_end", None) is not None:
        transforms_config["move_to_visit_end"] = args.move_to_visit_end

    if getattr(args, "move_to_visit_end", None) or transforms_config.get("move_to_visit_end"):
        visit_end_config = transforms_config.get("visit_end_time_config", {})
        if getattr(args, "visit_end_code_prefixes", None) is not None:
            visit_end_config["code_prefixes"] = args.visit_end_code_prefixes
        if getattr(args, "visit_end_check_description", None) is not None:
            visit_end_config["check_description"] = args.visit_end_check_description
        if getattr(args, "visit_code_pattern", None) is not None:
            visit_end_config["visit_code_pattern"] = args.visit_code_pattern
        if getattr(args, "workflow_stage_start", None) is not None:
            visit_end_config["workflow_stage_start"] = args.workflow_stage_start
        if getattr(args, "workflow_stage_end", None) is not None:
            visit_end_config["workflow_stage_end"] = args.workflow_stage_end
        transforms_config["visit_end_time_config"] = visit_end_config

    if args.convert_to_meds_reader is not None:
        transforms_config["convert_to_meds_reader"] = args.convert_to_meds_reader
    if args.meds_reader_output_dir is not None:
        transforms_config["meds_reader_output_dir"] = args.meds_reader_output_dir
    if args.verify_meds_reader is not None:
        transforms_config["verify_meds_reader"] = args.verify_meds_reader

    config["transforms"] = transforms_config

    if "input_dir" not in config:
        logger.error("Input directory is required (--input-dir or in config file)")
        sys.exit(1)

    if "output_dir" not in config:
        logger.error("Output directory is required (--output-dir or in config file)")
        sys.exit(1)

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])

    logger.info(f"Transforming files from {input_dir} to {output_dir}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Dataset metadata: {dataset_name} v{dataset_version}")

    try:
        validated_config = validate_transform_config(transforms_config)
    except Exception as e:
        logger.error(f"Invalid transform configuration: {e}")
        sys.exit(1)

    pipeline = MEDSTransformPipeline(
        config=validated_config,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    pipeline.transform_parquet_files(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    logger.info("Transformation completed successfully!")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Apply MEDS transformations to parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration file argument
    parser.add_argument("config", nargs="?", type=str, help="Path to YAML/JSON configuration file (optional)")

    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input-dir", type=str, help="Directory containing input MEDS data")
    io_group.add_argument("--output-dir", type=str, help="Directory to save transformed MEDS data")

    # Transform options
    transform_group = parser.add_argument_group("Transform Options")
    transform_group.add_argument(
        "--move-to-visit-end",
        action="store_true",
        default=None,
        help="Move events from visit start to visit end time for specified prefixes (fixes data leakage)",
    )
    transform_group.add_argument(
        "--no-move-to-visit-end",
        dest="move_to_visit_end",
        action="store_false",
        help="Disable visit end time moving transform",
    )

    # Visit end time options (uses workflow_stage column)
    visit_end_group = parser.add_argument_group("Visit End Time Options")
    visit_end_group.add_argument(
        "--visit-end-code-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Code prefixes to move to visit end (e.g., DIAGNOSIS/ PROCEDURE/ OBSERVATION/)",
    )
    visit_end_group.add_argument(
        "--visit-end-check-description",
        action="store_true",
        default=None,
        help="Also check description field for prefixes (default: True)",
    )
    visit_end_group.add_argument(
        "--no-visit-end-check-description",
        dest="visit_end_check_description",
        action="store_false",
        help="Only check code field for prefixes",
    )
    visit_end_group.add_argument(
        "--visit-code-pattern",
        type=str,
        default=None,
        help="Pattern to identify visit events in codes (default: VISIT)",
    )
    visit_end_group.add_argument(
        "--workflow-stage-start",
        type=str,
        default=None,
        help="Value in workflow_stage column indicating visit start (default: start, case-insensitive)",
    )
    visit_end_group.add_argument(
        "--workflow-stage-end",
        type=str,
        default=None,
        help="Value in workflow_stage column indicating visit end (default: end, case-insensitive)",
    )

    # MEDS Reader conversion options
    meds_reader_group = parser.add_argument_group("MEDS Reader Conversion Options")
    meds_reader_group.add_argument(
        "--convert-to-meds-reader",
        action="store_true",
        default=None,
        help="Convert output to MEDS Reader format (default: True)",
    )
    meds_reader_group.add_argument(
        "--no-convert-to-meds-reader",
        dest="convert_to_meds_reader",
        action="store_false",
        help="Skip conversion to MEDS Reader format",
    )
    meds_reader_group.add_argument(
        "--meds-reader-output-dir",
        type=str,
        help="Output directory for MEDS Reader dataset (default: {output_dir}_meds_reader)",
    )
    meds_reader_group.add_argument(
        "--verify-meds-reader",
        action="store_true",
        default=None,
        help="Verify MEDS Reader dataset after conversion (default: True)",
    )
    meds_reader_group.add_argument(
        "--no-verify-meds-reader",
        dest="verify_meds_reader",
        action="store_false",
        help="Skip MEDS Reader verification",
    )

    # Dataset metadata arguments
    metadata_group = parser.add_argument_group("Dataset Metadata")
    metadata_group.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset (e.g., 'My Dataset', default: 'Transformed Dataset')",
    )
    metadata_group.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Version of the dataset (e.g., '1.0', default: '1.0')",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    transform_command(args)


if __name__ == "__main__":
    main()
