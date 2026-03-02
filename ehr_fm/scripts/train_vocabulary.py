import argparse
from datetime import datetime

from torch.utils.data import DataLoader

from ehr_fm.data import create_dataset
from ehr_fm.logger import setup_logging
from ehr_fm.types import EventSequence
from ehr_fm.vocabulary import FactorizedVocab, JointVocab, QuantilePreScanner, Vocab


def parse_args():
    ap = argparse.ArgumentParser(description="Train vocabulary from MEDS data")

    # Dataset arguments
    ap.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")

    # Common arguments
    ap.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to samples.parquet file. If not provided, uses metadata/samples.parquet",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for vocabulary files")
    ap.add_argument(
        "--split", type=str, default=None, help="Data split to use (train, val, test, validation)"
    )
    ap.add_argument("--vocab_size", type=int, default=98_304, help="Maximum vocabulary size")
    ap.add_argument(
        "--numeric_sample_reservoir_size",
        type=int,
        default=1000,
        help="Size of reservoir for numeric value sampling",
    )
    ap.add_argument("--seed", type=int, default=444, help="Random seed for reproducibility")
    ap.add_argument("--save_reservoirs", action="store_true", help="Save reservoir sampling data")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing vocabulary files")

    # Interval token options
    ap.add_argument(
        "--add_time_interval_tokens",
        action="store_true",
        help=("Prepend time interval tokens to the vocabulary to ensure inclusion and stable IDs"),
    )
    ap.add_argument(
        "--add_demographic_tokens",
        action="store_true",
        help=("Prepend demographic tokens (sex, age buckets, first-event year buckets) to the vocabulary"),
    )
    ap.add_argument(
        "--numeric_bin_by_unit",
        action="store_true",
        help="Enable unit-specific binning for numeric values",
    )
    ap.add_argument(
        "--min_samples_per_unit",
        type=int,
        default=None,
        help="Minimum number of samples required per (code, unit) pair for binning",
    )

    # Factorized vocabulary arguments
    ap.add_argument(
        "--tokenization_mode",
        type=str,
        choices=["joint", "factorized"],
        default="joint",
        help="Vocabulary mode: 'joint' (code-specific bins) or 'factorized' (generic Q:*/STAGE:* tokens)",
    )
    ap.add_argument(
        "--num_quantiles",
        type=int,
        default=10,
        help="Number of quantile buckets for factorized mode (default: 10)",
    )
    ap.add_argument(
        "--emit_quantiles",
        action="store_true",
        default=True,
        help="Include Q:* tokens in factorized vocabulary (default: True)",
    )
    ap.add_argument(
        "--no_emit_quantiles",
        action="store_true",
        help="Exclude Q:* tokens from factorized vocabulary",
    )
    ap.add_argument(
        "--emit_text",
        action="store_true",
        default=True,
        help="Include TXT:* tokens in factorized vocabulary (default: True)",
    )
    ap.add_argument(
        "--no_emit_text",
        action="store_true",
        help="Exclude TXT:* tokens from factorized vocabulary",
    )
    ap.add_argument(
        "--emit_stage",
        action="store_true",
        default=True,
        help="Include STAGE:* tokens in factorized vocabulary (discovers from data)",
    )
    ap.add_argument(
        "--no_emit_stage",
        action="store_true",
        help="Disable STAGE:* token emission in factorized vocabulary",
    )
    ap.add_argument(
        "--remove_prefixes",
        action="store_true",
        default=True,
        help="Remove code prefixes in factorized mode (e.g., LAB/glucose → glucose)",
    )
    ap.add_argument(
        "--no_remove_prefixes",
        action="store_true",
        help="Keep full code strings in factorized vocabulary",
    )
    ap.add_argument(
        "--code_separator",
        type=str,
        default="/",
        help="Code separator for prefix removal (default: '/')",
    )

    return ap.parse_args()


def custom_collate(
    batch: list[tuple[tuple[int, datetime], EventSequence] | None]
) -> tuple[list[tuple[int, datetime]], list[EventSequence]]:
    filtered_batch = [item for item in batch if item is not None]

    if not filtered_batch:
        return [], []

    ids, data = zip(*filtered_batch)
    return list(ids), list(data)


def main():
    logger = setup_logging(child_name="train_vocabulary")

    args = parse_args()

    dataset_config = {
        "split": args.split,
        "dataset_path": args.dataset_path,
    }

    logger.info(f"Using dataset with auto-detection: {args.dataset_path}")

    dataset_config["samples_path"] = args.samples_path

    logger.info(f"Creating dataset with config: {dataset_config}")
    dataset = create_dataset(dataset_config)
    logger.info(f"Created {type(dataset).__name__} with {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        collate_fn=custom_collate,
    )

    vocab_config = {
        "vocab_size": args.vocab_size,
        "numeric_sample_reservoir_size": args.numeric_sample_reservoir_size,
        "seed": args.seed,
        "n_samples": len(dataset),
        "numeric_bin_by_unit": args.numeric_bin_by_unit,
        "min_samples_per_unit": args.min_samples_per_unit,
    }

    # Determine factorized/joint settings
    emit_quantiles = args.emit_quantiles and not args.no_emit_quantiles
    emit_text = args.emit_text and not args.no_emit_text
    emit_stage = args.emit_stage and not args.no_emit_stage
    remove_prefixes = args.remove_prefixes and not args.no_remove_prefixes

    # Pre-scan for quantile breaks if emit_quantiles=True (two-pass training)
    needs_prescan = args.tokenization_mode in ("factorized", "joint") and emit_quantiles

    quantile_breaks = {}
    discovered_stages = []

    if needs_prescan:
        logger.info("Starting two-pass vocabulary training...")
        logger.info("Pass 1: Pre-scanning for quantile breaks and stage discovery...")

        prescanner = QuantilePreScanner(
            num_quantiles=args.num_quantiles,
            reservoir_size=args.numeric_sample_reservoir_size,
            seed=args.seed,
            numeric_bin_by_unit=args.numeric_bin_by_unit,
        )

        # Create a fresh dataloader for pre-scan
        prescan_dataloader = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=False,
            collate_fn=custom_collate,
        )
        prescanner.scan(prescan_dataloader)
        quantile_breaks = prescanner.compute_breaks()
        discovered_stages = prescanner.get_discovered_stages()

        logger.info(
            f"Pre-scan complete: {len(quantile_breaks)} codes with breaks, "
            f"{len(discovered_stages)} stages discovered"
        )
        logger.info("Pass 2: Main vocabulary training with fixed breaks...")

        dataloader = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=False,
            collate_fn=custom_collate,
        )

    if args.tokenization_mode == "factorized":
        vocab = FactorizedVocab(
            vocab_config,
            num_quantiles=args.num_quantiles,
            emit_quantiles=emit_quantiles,
            emit_text=emit_text,
            emit_stage=emit_stage,
            remove_prefixes=remove_prefixes,
            separator=args.code_separator,
            precomputed_quantile_breaks=quantile_breaks if needs_prescan else None,
            precomputed_stages=discovered_stages if needs_prescan else None,
        )
        mode_str = "FACTORIZED"
        logger.info(
            f"Training {mode_str} vocabulary with config: {vocab_config}, "
            f"num_quantiles={args.num_quantiles}, emit_quantiles={emit_quantiles}, "
            f"emit_text={emit_text}, emit_stage={emit_stage}, "
            f"remove_prefixes={remove_prefixes}"
        )
        vocab.train(dataloader=dataloader)

    elif args.tokenization_mode == "joint":
        vocab = JointVocab(
            vocab_config,
            quantile_breaks=quantile_breaks,
            discovered_stages=discovered_stages,
            num_quantiles=args.num_quantiles,
            emit_quantiles=emit_quantiles,
            emit_text=emit_text,
            emit_stage=emit_stage,
            remove_prefixes=remove_prefixes,
            separator=args.code_separator,
        )
        mode_str = "JOINT"
        logger.info(
            f"Training {mode_str} vocabulary with combined tokens: {vocab_config}, "
            f"num_quantiles={args.num_quantiles}, emit_quantiles={emit_quantiles}, "
            f"emit_text={emit_text}, emit_stage={emit_stage}, "
            f"remove_prefixes={remove_prefixes}"
        )
        vocab.train(dataloader=dataloader)

    else:
        vocab = Vocab(vocab_config)
        logger.info(f"Training LEGACY JOINT vocabulary with config: {vocab_config}")
        vocab.train(dataloader=dataloader)

    # Optionally prepend interval tokens before saving (to avoid truncation)
    if args.add_time_interval_tokens or args.add_demographic_tokens:
        from ehr_fm.vocabulary import (
            prepend_demographic_tokens,
            prepend_interval_tokens,
        )

        # Get vocab - handle both old (list) and new (tuple) return formats
        vocab_result = vocab.get_vocab()
        if isinstance(vocab_result, tuple):
            vocab_entries, num_ntp_classes = vocab_result
        else:
            vocab_entries = vocab_result
            num_ntp_classes = len(vocab_entries)

        if args.add_demographic_tokens:
            vocab_entries = prepend_demographic_tokens(
                vocab_entries,
                target_vocab_size=getattr(vocab, "vocab_size", None),
            )
        if args.add_time_interval_tokens:
            vocab_entries = prepend_interval_tokens(
                vocab_entries,
                target_vocab_size=getattr(vocab, "vocab_size", None),
            )

        # Re-package as tuple if needed
        if isinstance(vocab, (JointVocab, FactorizedVocab)):
            vocab_entries = (vocab_entries, num_ntp_classes)

    else:
        vocab_entries = None  # signal to save() to call get_vocab internally

    vocab.save(
        args.out_dir,
        save_reservoirs=args.save_reservoirs,
        overwrite=args.overwrite,
        precomputed_vocab=vocab_entries,
    )
    logger.info(f"Vocabulary saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
