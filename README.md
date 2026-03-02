# EHR-FM

A Python package for training foundation models on electronic health record data in [MEDS Reader](https://github.com/som-shahlab/meds_reader) format.

## Requirements

- Python >= 3.10, \< 3.12
- PyTorch ~2.7
- Transformers ~4.51
- xformers 0.0.30
- CUDA-capable GPU (recommended for training)

Key data dependencies:

- [meds-reader](https://github.com/som-shahlab/meds_reader) 0.1.9
- [meds](https://github.com/Medical-Event-Data-Standard/meds) 0.3.3
- polars ~1.27

See `pyproject.toml` for the full dependency list.

## Installation

```bash
cd ehr-fm
pip install -e .
```

## Data Format

EHR-FM requires input data in **MEDS Reader** format. If your data is in standard MEDS format, convert it first using one of the methods below.

### Converting MEDS to MEDS Reader

**CLI** (provided by the `meds-reader` package):

```bash
meds_reader_convert /path/to/meds_dataset /path/to/meds_reader_output
```

**Python** (via `ehr_fm.meds_reader_utils`):

```python
from pathlib import Path
from ehr_fm.meds_reader_utils import convert_to_meds_reader, verify_meds_reader

convert_to_meds_reader(
    Path("/path/to/meds_dataset"), Path("/path/to/meds_reader_output")
)

# Optional: verify the conversion
verify_meds_reader(Path("/path/to/meds_dataset"), Path("/path/to/meds_reader_output"))
```

The `apply_meds_transforms` command can also convert its output to MEDS Reader format via the `--convert-to-meds-reader` flag (enabled by default).

## Pipeline Overview

The typical workflow has five steps. See [`ehr_fm/scripts/README.md`](ehr_fm/scripts/README.md) for detailed CLI documentation and examples.

1. **Preprocessing** (`apply_meds_transforms`) -- Optional. Fix data-leakage issues such as events timestamped at admission instead of discharge (in MIMIC).
2. **Vocabulary** (`train_vocabulary`) -- Build a token vocabulary from the dataset. Supports joint and factorized tokenization modes.
3. **Tokenization** (`pretokenize`) -- Convert events to token sequences using the trained vocabulary.
4. **Training** (`train_ehr_fm`) -- Train the model on tokenized data. Requires a YAML/JSON config file as the first positional argument; run `train_ehr_fm --help` for full usage. See the [CLI docs](ehr_fm/scripts/README.md#train_ehr_fm) for boolean flag semantics and checkpoint resumption options.
5. **Representations** (`compute_representations`) -- Extract patient embeddings from a trained model.

## Further Reading

- [Tutorial](tutorials/tutorial.ipynb) -- End-to-end notebook walking through vocabulary training, tokenization, model training, and representation extraction on synthetic data
- [CLI Documentation](ehr_fm/scripts/README.md) -- Detailed usage, arguments, and examples for all commands
- [Data Module](ehr_fm/data/README.md) -- Dataset classes and configuration options
