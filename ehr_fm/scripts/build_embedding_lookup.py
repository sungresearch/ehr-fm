import argparse

from ehr_fm.embedding import EmbeddingLookupConfig, build_embedding_lookup
from ehr_fm.logger import setup_logging


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a frozen embedding lookup table from MEDS embedding_text strings",
    )
    ap.add_argument(
        "--meds_dir",
        type=str,
        required=True,
        help="Path to MEDS dataset root (must contain data/*.parquet with embedding_text column)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for embedding artifacts",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace model identifier for the text embedding model (default: Qwen/Qwen3-Embedding-8B)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for GPU inference (default: 64)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding model inference (default: cuda)",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage precision for embedding table (default: float16)",
    )
    return ap.parse_args()


def main():
    logger = setup_logging(child_name="build_embedding_lookup")
    args = parse_args()

    config = EmbeddingLookupConfig(
        meds_dir=args.meds_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    logger.info(f"Config: {config.model_dump()}")
    build_embedding_lookup(config)


if __name__ == "__main__":
    main()
