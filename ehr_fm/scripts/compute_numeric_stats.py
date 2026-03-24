"""Compute per-code numeric statistics for the 5-dim numerical feature vector.

Scans MEDS Reader data to precompute:
- log-transform z-score statistics (log_mean, log_std) per code
- quantile breaks per code (via QuantilePreScanner)
"""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from ehr_fm.data import create_dataset
from ehr_fm.logger import setup_logging
from ehr_fm.types import EventSequence
from ehr_fm.vocabulary import QuantilePreScanner


def _collate(
    batch: list[tuple[tuple[int, datetime], EventSequence] | None],
) -> tuple[list[tuple[int, datetime]], list[EventSequence]]:
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return [], []
    ids, data = zip(*filtered)
    return list(ids), list(data)


def main():
    parser = argparse.ArgumentParser(description="Compute per-code numeric statistics for embedding mode.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MEDS Reader dataset.")
    parser.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to samples.parquet. Defaults to {dataset_path}/metadata/samples.parquet.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path for numeric_stats.json.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Data split to scan (train, val, test, validation). Default: all samples.",
    )
    parser.add_argument(
        "--num_quantiles", type=int, default=10, help="Number of quantile buckets (default: 10)."
    )
    parser.add_argument(
        "--numeric_sample_reservoir_size",
        type=int,
        default=1000,
        help="Reservoir size for numeric value sampling per code (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=444, help="Random seed for reservoir sampling.")
    args = parser.parse_args()

    logger = setup_logging(child_name="compute_numeric_stats")

    dataset_config = {
        "dataset_path": args.dataset_path,
        "samples_path": args.samples_path,
        "split": args.split,
    }
    dataset = create_dataset(dataset_config)
    logger.info(f"Dataset loaded: {len(dataset)} samples (split={args.split})")

    prescanner = QuantilePreScanner(
        num_quantiles=args.num_quantiles,
        reservoir_size=args.numeric_sample_reservoir_size,
        seed=args.seed,
    )

    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, collate_fn=_collate)

    code_log_values: dict[str, list[float]] = defaultdict(list)

    logger.info("Scanning data for log-stats and quantile breaks (single pass)...")
    for _, sequences in dataloader:
        prescanner.forward(sequences)
        for events in sequences:
            for event in events:
                numeric_value = event.get("numeric_value")
                if numeric_value is not None:
                    code_log_values[event["code"]].append(math.log1p(abs(numeric_value)))

    quantile_breaks = prescanner.compute_breaks()

    output = {}
    for code, vals in code_log_values.items():
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 1.0
        if std < 1e-8:
            std = 1.0
        entry = {
            "log_mean": mean,
            "log_std": std,
            "n_samples": n,
        }
        if code in quantile_breaks:
            entry["quantile_breaks"] = quantile_breaks[code]
        output[code] = entry

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        f"Wrote numeric stats for {len(output)} codes "
        f"({sum(1 for v in output.values() if 'quantile_breaks' in v)} with quantile breaks) "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
