import argparse
import datetime
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ehr_fm.data import TokenBudgetBatchSampler, TokenizedDataset
from ehr_fm.logger import setup_logging
from ehr_fm.models.transformer import EHRFM, packed_ehr_collate
from ehr_fm.tokenizer import pretokenize_data


def create_representations(
    tokenized_file_path: Path,
    model: EHRFM,
    device: torch.device,
    output_path: Path,
    max_tokens_per_batch: int,
    min_patients_per_batch: int,
    *,
    shuffle_batches: bool = True,
    loader_num_workers: int = 0,
    convert_ages_to_positions: bool = False,
    logger,
) -> None:
    """Create patient representations from a tokenized dataset file."""
    logger.info("Loading TokenizedDataset...")

    # This returns one window per patient, sampling up to (max_tokens_per_batch // min_patients_per_batch)
    # tokens per patient. For example, if max_tokens_per_batch = 16384 and min_patients_per_batch = 1, then
    # this will sample up to 16384 tokens per patient. Note that sampling starts at the end of the patient's
    # sequence and works backwards
    dataset = TokenizedDataset(
        tokenized_file_path,
        max_length=max_tokens_per_batch // min_patients_per_batch,
        one_window=True,
        convert_ages_to_positions=convert_ages_to_positions,
    )

    sampler = TokenBudgetBatchSampler(
        dataset,
        tokens_per_batch=max_tokens_per_batch,
        min_patients=min_patients_per_batch,
        shuffle=shuffle_batches,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=loader_num_workers,
        persistent_workers=(loader_num_workers > 0),
        pin_memory=True,
        collate_fn=packed_ehr_collate,
    )

    all_representations = []
    all_patient_ids = []
    all_index_times = []

    logger.info("Starting representation computation...")

    # Get model's dtype from embedding layer
    if hasattr(model.transformer, "embed_bag"):
        model_dtype = model.transformer.embed_bag.weight.dtype
    else:
        model_dtype = model.transformer.embed.weight.dtype

    for batch in tqdm(dataloader, desc="Computing representations"):
        transformer_input_on_device = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Convert float tensors to model dtype, keep integer tensors as-is
                if value.dtype.is_floating_point:
                    transformer_input_on_device[key] = value.to(device=device, dtype=model_dtype)
                else:
                    transformer_input_on_device[key] = value.to(device)
            else:
                transformer_input_on_device[key] = value

        with torch.no_grad():
            hidden_states = model.transformer(transformer_input_on_device)

        patient_lengths = transformer_input_on_device["patient_lengths"]
        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_level_representations = hidden_states[end_indices]

        all_representations.append(patient_level_representations.cpu())
        all_patient_ids.extend(batch["patient_ids"].tolist())
        all_index_times.extend(batch["index_times"].tolist())

    if not all_representations:
        logger.warning("No representations were computed. Check input data and dataloader.")
        return

    final_representations = torch.cat(all_representations, dim=0)
    logger.info(
        f"Computed {len(final_representations)} representations for "
        f"{len(all_patient_ids)} unique patients/sequences."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving representations to: {output_path}")

    df_representations = pl.DataFrame(
        {
            "id": [str(x) for x in all_patient_ids],
            "index_t": [datetime.datetime.fromtimestamp(int(x)) for x in all_index_times],
            "representations": [np.array(x) for x in final_representations.tolist()],
        }
    )

    df_representations.write_parquet(output_path)
    logger.info("Representations saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute patient representations using a EHRFM model. "
            "This script can either pretokenize MEDS data into a temporary Parquet file or use "
            "an existing pretokenized dataset, then loads this tokenized data, computes representations "
            "using the specified EHRFM model, and finally saves the representations (patient ID, index time, "
            "and representation vector) to a Parquet file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example usage:\n"
            "  # Using raw MEDS data (will pretokenize):\n"
            "  python compute_representations.py \\\n"
            "    --dataset_path /path/to/your/meds_reader_data \\\n"
            "    --samples_path /path/to/your/samples.parquet \\\n"
            "    --model_path /path/to/your/clmbr_model_checkpoint \\\n"
            "    --vocab_path /path/to/your/vocab.json \\\n"
            "    --output_path /path/to/save/representations.parquet \\\n"
            "    --split validation \\\n"
            "    --num_workers 4 \\\n"
            "    --max_tokens_per_batch 16384 \\\n"
            "    --min_patients_per_batch 1\n"
            "\n"
            "  # Using pretokenized data (skips pretokenization):\n"
            "  python compute_representations.py \\\n"
            "    --pretokenized_dataset_path /path/to/pretokenized_patients.parquet \\\n"
            "    --model_path /path/to/your/clmbr_model_checkpoint \\\n"
            "    --vocab_path /path/to/your/vocab.json \\\n"
            "    --output_path /path/to/save/representations.parquet \\\n"
            "    --max_tokens_per_batch 16384 \\\n"
            "    --min_patients_per_batch 1\n"
            "\n"
            "Notes:\n"
            "- Either --dataset_path or --pretokenized_dataset_path must be provided (mutually exclusive).\n"
            "- If --samples_path is not provided with --dataset_path, it defaults to "
            "{data_path}/metadata/samples.parquet.\n"
            "- The script handles temporary storage of pretokenized data automatically when using --dataset_path.\n"
            "- TokenizedDataset samples one window (up to max_tokens_per_batch / min_patients_per_batch tokens) "
            "from the end of each patient sequence."
        ),
    )

    # Dataset type arguments (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_path", type=str, help="Path to the MEDS reader data directory (legacy)."
    )
    dataset_group.add_argument(
        "--pretokenized_dataset_path",
        type=str,
        help="Path to an existing pretokenized dataset Parquet file. Skips pretokenization step.",
    )

    # Common arguments
    parser.add_argument(
        "--samples_path",
        type=str,
        default=None,
        help="Path to the samples.parquet file. Defaults to {data_path}/metadata/samples.parquet.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained EHRFM model.")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to the vocabulary file used for pretokenization.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the computed representations (e.g., output.pt or output_dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional data split to process (e.g., 'train', 'validation', 'test').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of workers for pretokenization (0 for single-process, -1 for all cores).",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=16_384,
        help="Maximum number of tokens to process per batch.",
    )
    parser.add_argument(
        "--min_patients_per_batch", type=int, default=1, help="Minimum number of patients per batch."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Max vocab size. Defaults to length of vocabulary",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 precision for representation computation.",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        help="Use BFloat16 precision for representation computation.",
    )

    parser.add_argument(
        "--convert_ages_to_positions",
        action="store_true",
        help="Convert ages to sequential positions at runtime. Must match training configuration.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info(f"Loading EHRFM model from: {args.model_path}")

    torch_dtype = None
    if args.use_fp16:
        torch_dtype = torch.float16
    elif args.use_bfloat16:
        torch_dtype = torch.bfloat16

    model = EHRFM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.pretokenized_dataset_path:
        tokenized_file_path = Path(args.pretokenized_dataset_path)
        logger.info(f"Using pretokenized dataset from: {tokenized_file_path}")
        create_representations(
            tokenized_file_path=tokenized_file_path,
            model=model,
            device=device,
            output_path=Path(args.output_path),
            max_tokens_per_batch=args.max_tokens_per_batch,
            min_patients_per_batch=args.min_patients_per_batch,
            shuffle_batches=False,
            loader_num_workers=max(0, args.num_workers),
            convert_ages_to_positions=args.convert_ages_to_positions,
            logger=logger,
        )
    else:
        if args.vocab_path is None:
            raise ValueError("vocab_path is required for pretokenization when using dataset_path")

        data_path = Path(args.dataset_path)
        pretokenized_parquet_filename = "pretokenized_patients.parquet"

        if args.samples_path:
            samples_path = Path(args.samples_path)
        else:
            samples_path = data_path / "metadata" / "samples.parquet"
            logger.info(f"Using default samples_path: {samples_path}")

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path = Path(temp_dir_str)
            logger.info(f"Created temporary directory for pretokenized data: {temp_dir_path}")

            logger.info("Starting pretokenization...")
            pretokenize_data(
                vocab_path=args.vocab_path,
                out_dir=temp_dir_path,
                dataset_path=args.dataset_path,
                samples_path=samples_path,
                split=args.split,
                num_workers=args.num_workers,
                output_filename=pretokenized_parquet_filename,
                vocab_size=args.vocab_size,
            )

            tokenized_file_path = temp_dir_path / pretokenized_parquet_filename
            logger.info(f"Pretokenized data written to: {tokenized_file_path}")

            # Process representations while the temporary directory still exists
            create_representations(
                tokenized_file_path=tokenized_file_path,
                model=model,
                device=device,
                output_path=Path(args.output_path),
                max_tokens_per_batch=args.max_tokens_per_batch,
                min_patients_per_batch=args.min_patients_per_batch,
                shuffle_batches=False,
                loader_num_workers=max(0, args.num_workers),
                convert_ages_to_positions=args.convert_ages_to_positions,
                logger=logger,
            )


if __name__ == "__main__":
    main()
