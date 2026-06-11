# Data Module (Datasets)

Datasets, batch sampling, and collation for EHR-FM. All public names are exported from `ehr_fm.data`:

```python
from ehr_fm.data import (
    create_dataset,
    MEDSReaderDataset,
    MEDSReaderDatasetConfig,
    TokenizedDataset,
    TokenBudgetBatchSampler,
    packed_ehr_collate,
)
```

## Quick Start

```python
from ehr_fm.data import create_dataset

# dataset_path points to a MEDS Reader directory.
config = {"dataset_path": "/path/to/meds_reader_data", "split": "train"}
dataset = create_dataset(config)
```

## Supported Data Format

The package requires data in **MEDS Reader** format.

- **Structure**: directories (`code/`, `description/`, `numeric_value/`, `time/`) and files (`subject_id`, `meds_reader.length`, `meds_reader.properties`, `meds_reader.version`).
- **Dataset class**: `MEDSReaderDataset`.

`create_dataset()` validates the directory structure and returns a `MEDSReaderDataset`.

## Configuration Options

`create_dataset` accepts a dict (validated by `MEDSReaderDatasetConfig`):

| Parameter      | Type       | Description                                           |
| -------------- | ---------- | ----------------------------------------------------- |
| `dataset_path` | `str`      | Path to MEDS Reader dataset directory (required)      |
| `samples_path` | `str`      | Path to samples parquet file (optional)               |
| `split`        | `str`      | Data split to load: "train", "val", "test" (optional) |
| `transform`    | `callable` | Function to transform samples (optional)              |

## TokenizedDataset

For sequence modeling over pretokenized patient records (output of `pretokenize` or `pretokenize_embedding`):

```python
from ehr_fm.data import TokenizedDataset

dataset = TokenizedDataset(
    parquet_path="/path/to/tokenized.parquet",
    max_length=2048,
    one_window=True,  # single end-anchored window per patient
    dropout_prob=0.1,  # per-window token dropout
)
```

`one_window=False` yields multiple sliding windows per patient (stepping backward from the most recent event). Optional `embedding_text_ids` / `numeric_features` columns are surfaced automatically when present in the Parquet.

## Batching & Collation

Training batches are formed by **token budget**, not a fixed batch size, and packed flat for block-diagonal attention:

```python
from torch.utils.data import DataLoader
from ehr_fm.data import TokenBudgetBatchSampler, packed_ehr_collate

sampler = TokenBudgetBatchSampler(dataset, tokens_per_batch=16384, min_patients=2)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=packed_ehr_collate)
```

- **`TokenBudgetBatchSampler`** groups samples so each batch stays near `tokens_per_batch` (while honoring `min_patients`).
- **`packed_ehr_collate`** concatenates variable-length windows into one flat token sequence and records `patient_lengths`, so the transformer applies a block-diagonal mask (no cross-patient attention).

This is the collation used by `FMTrainer`; see [`ehr_fm/scripts/README.md`](../scripts/README.md#train_ehr_fm) for the training CLI.
