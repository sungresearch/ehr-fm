"""Compute per-code numeric statistics for the 5-dim numerical feature vector.

Scans MEDS Reader data to precompute log-transform z-score statistics
(log_mean, log_std) per code. Output: numeric_stats.json with per-code
{log_mean, log_std, n_samples}.
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


def _collate(
    batch: list[tuple[tuple[int, datetime], EventSequence] | None],
) -> tuple[list[tuple[int, datetime]], list[EventSequence]]:
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return [], []
    ids, data = zip(*filtered)
    return list(ids), list(data)


def compute_log_stats(code_log_values: dict[str, list[float]]) -> dict[str, dict]:
    """Per-code log-transform statistics: {log_mean, log_std, n_samples}.

    log_std falls back to 1.0 when a code has <=1 sample or a degenerate
    (<1e-8) spread. Codes with no samples are omitted.
    """
    output: dict[str, dict] = {}
    for code, vals in code_log_values.items():
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 1.0
        if std < 1e-8:
            std = 1.0
        output[code] = {"log_mean": mean, "log_std": std, "n_samples": n}
    return output


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
    args = parser.parse_args()

    logger = setup_logging(child_name="compute_numeric_stats")

    dataset_config = {
        "dataset_path": args.dataset_path,
        "samples_path": args.samples_path,
        "split": args.split,
    }
    dataset = create_dataset(dataset_config)
    logger.info(f"Dataset loaded: {len(dataset)} samples (split={args.split})")

    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, collate_fn=_collate)

    code_log_values: dict[str, list[float]] = defaultdict(list)

    logger.info("Scanning data for log-stats...")
    for _, sequences in dataloader:
        for events in sequences:
            for event in events:
                numeric_value = event.get("numeric_value")
                if numeric_value is not None:
                    code_log_values[event["code"]].append(math.log1p(abs(numeric_value)))

    output = compute_log_stats(code_log_values)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote numeric stats for {len(output)} codes to {out_path}")


if __name__ == "__main__":
    main()
