"""CLI for embedding-mode pretokenization.

Thin argparse wrapper over ``ehr_fm.pretokenize.pretokenize_embedding_data``.
See that module (and ehr_fm.pretokenize.embedding_*) for the pipeline itself.
"""

import argparse

from ehr_fm.pretokenize import pretokenize_embedding_data


def main():
    parser = argparse.ArgumentParser(description="Embedding-mode pretokenization.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MEDS Reader dataset.")
    parser.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to samples.parquet.",
    )
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json.")
    parser.add_argument(
        "--embedding_lookup_path", type=str, required=True, help="Path to embedding lookup artifacts dir."
    )
    parser.add_argument(
        "--numeric_stats_path", type=str, default=None, help="Path to numeric_stats.json (optional)."
    )
    parser.add_argument(
        "--numeric_pathway_mode",
        type=str,
        choices=["legacy_zscore", "ref_range_priority"],
        default="legacy_zscore",
        help="Numeric feature construction mode. 'legacy_zscore': 5-dim vector using numeric_stats.json. "
        "'ref_range_priority': 4-dim institution-invariant vector (no external stats required).",
    )
    parser.add_argument(
        "--numeric_override_mode",
        type=str,
        choices=["none", "zero"],
        default="none",
        help="'none': standard numeric features. 'zero': force z-score=0, quantile=0.5 for all events. "
        "Only applicable under legacy_zscore pathway mode.",
    )
    parser.add_argument(
        "--numeric_quantile_breaks_path",
        type=str,
        default=None,
        help="Override quantile_breaks source for numeric features. Accepts a vocab.json "
        "(extracts quantile_breaks key) or a bare {code: [breaks]} JSON dict. "
        "Only applicable under legacy_zscore pathway mode.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--split", type=str, default=None, help="Data split (default: all splits).")
    parser.add_argument("--workers", type=int, default=-1, help="Number of workers (-1 = all cores).")
    parser.add_argument("--vocab_size", type=int, default=None, help="Max vocab size for NTP labels.")
    parser.add_argument(
        "--row_group_size", type=int, default=32768, help="Row group size for parquet output."
    )
    args = parser.parse_args()

    pretokenize_embedding_data(
        vocab_path=args.vocab_path,
        embedding_lookup_path=args.embedding_lookup_path,
        out_dir=args.out_dir,
        dataset_path=args.dataset_path,
        samples_path=args.samples_path,
        split=args.split,
        numeric_stats_path=args.numeric_stats_path,
        numeric_pathway_mode=args.numeric_pathway_mode,
        numeric_override_mode=args.numeric_override_mode,
        numeric_quantile_breaks_path=args.numeric_quantile_breaks_path,
        num_workers=args.workers,
        vocab_size=args.vocab_size,
        row_group_size=args.row_group_size,
    )


if __name__ == "__main__":
    main()
