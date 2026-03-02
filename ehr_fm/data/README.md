# Data Module (Datasets)

## Quick Start

```python
from ehr_fm.data.dataset import create_dataset

# Specify dataset_path pointing to a MEDS Reader directory.
config = {"dataset_path": "/path/to/meds_reader_data", "split": "train"}
dataset = create_dataset(config)
```

## Supported Data Format

The package requires data in **MEDS Reader** format.

### MEDS Reader Format

- **Structure**: Directories (`code/`, `description/`, `numeric_value/`, `time/`) and files (`subject_id`, `meds_reader.length`, `meds_reader.properties`, `meds_reader.version`)
- **Dataset Class**: `MEDSReaderDataset`

The `create_dataset()` factory function validates the directory structure and returns a `MEDSReaderDataset`.

## Configuration Options

| Parameter      | Type       | Description                                           |
| -------------- | ---------- | ----------------------------------------------------- |
| `dataset_path` | `str`      | Path to MEDS Reader dataset directory (required)      |
| `samples_path` | `str`      | Path to samples parquet file (optional)               |
| `split`        | `str`      | Data split to load: "train", "val", "test" (optional) |
| `transform`    | `callable` | Function to transform samples (optional)              |

## Specialized Datasets

### TokenizedDataset

For sequence modeling with tokenized patient records:

```python
from ehr_fm.data.dataset import TokenizedDataset

dataset = TokenizedDataset(
    parquet_path="/path/to/tokenized.parquet",
    max_length=2048,
    one_window=True,
    dropout_prob=0.1,
)
```
