import argparse

from ehr_fm.tokenizer import pretokenize_data


def parse_args():
    ap = argparse.ArgumentParser(description="Pretokenize MEDS data")

    # Dataset type arguments (mutually exclusive)
    dataset_group = ap.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset_path", type=str, help="Path to dataset")

    # Common arguments
    ap.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file (JSON/YAML)")
    ap.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to samples.parquet file. If not provided, uses metadata/samples.parquet",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for tokenized files")
    ap.add_argument(
        "--split", type=str, default=None, help="Data split to use (train, val, test, validation)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of worker processes. 0 = single-process, -1 = all cores",
    )
    ap.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Max vocab size. Defaults to length of vocabulary",
    )
    ap.add_argument(
        "--max_events_per_patient",
        type=int,
        default=None,
        help="Maximum number of events to process per patient. If not specified, all events are processed",
    )
    ap.add_argument(
        "--row_group_size",
        type=int,
        default=32_768,
        help="Number of rows per group in the output Parquet file",
    )
    ap.add_argument(
        "--output_filename",
        type=str,
        default="patients_tokenized.parquet",
        help="Name of the output Parquet file",
    )

    # Time-interval and demographic arguments
    ap.add_argument(
        "--inject_time_intervals",
        action="store_true",
        help="Inject temporal interval tokens between events",
    )
    ap.add_argument(
        "--demographic_prefix",
        action="store_true",
        help="Prepend demographic tokens and start timeline at first event",
    )
    ap.add_argument(
        "--demographic_include_year",
        action="store_true",
        help="Include year token in demographic prefix (only used when --demographic_prefix is enabled)",
    )
    ap.add_argument(
        "--sex_codes_male",
        type=str,
        default=None,
        help="Comma-separated code strings to recognize as male sex",
    )
    ap.add_argument(
        "--sex_codes_female",
        type=str,
        default=None,
        help="Comma-separated code strings to recognize as female sex",
    )
    ap.add_argument(
        "--sex_codes_unknown",
        type=str,
        default=None,
        help="Comma-separated code strings to recognize as unknown sex",
    )
    ap.add_argument(
        "--max_interval_repeat",
        type=int,
        default=None,
        help=(
            "Maximum number of repeatable 6-month interval tokens to insert. " "Defaults to None (unlimited)"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()

    pretokenize_data(
        vocab_path=args.vocab_path,
        out_dir=args.out_dir,
        dataset_path=args.dataset_path,
        samples_path=args.samples_path,
        split=args.split,
        num_workers=args.workers,
        max_events_per_patient=args.max_events_per_patient,
        row_group_size=args.row_group_size,
        output_filename=args.output_filename,
        vocab_size=args.vocab_size,
        inject_time_intervals=args.inject_time_intervals,
        max_interval_repeat=args.max_interval_repeat,
        demographic_prefix=args.demographic_prefix,
        demographic_include_year=args.demographic_include_year,
        sex_codes_male=args.sex_codes_male,
        sex_codes_female=args.sex_codes_female,
        sex_codes_unknown=args.sex_codes_unknown,
    )


if __name__ == "__main__":
    main()
