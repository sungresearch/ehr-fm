# EHR-FM

A Python package for training **embedding-first foundation models** on electronic health record data in [MEDS Reader](https://github.com/som-shahlab/meds_reader) format. Events are described in natural language, encoded with a frozen text-embedding model, and modeled by a transformer whose positions are calibrated to clinical time. The package supports two input modes — a **discrete** vocabulary path and an **embedding** path — from a shared preprocessing and vocabulary base.

## Requirements

- Python >= 3.10, \< 3.14
- PyTorch ~2.7
- Transformers ~4.51
- xformers 0.0.31
- CUDA-capable GPU (recommended for training)

Key data dependencies:

- [meds-reader](https://github.com/som-shahlab/meds_reader) 0.1.16
- [meds](https://github.com/Medical-Event-Data-Standard/meds) 0.3.3
- polars ~1.27

See `pyproject.toml` for the full dependency list.

## Installation

```bash
cd ehr-fm
pip install -e .
pip install xformers==0.0.31 --no-build-isolation
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

After optional preprocessing and vocabulary training (shared), the pipeline splits into two input modes. Pick the one that matches the model you want to train. See [`ehr_fm/scripts/README.md`](ehr_fm/scripts/README.md) for full CLI documentation and examples.

```
                      apply_meds_transforms        (optional: fix timestamp leakage)
                                │
                        train_vocabulary           → vocab.json
                                │
              ┌─────────────────┴───────────────────┐
       DISCRETE mode                          EMBEDDING mode
              │                                       │
        pretokenize                       build_embedding_lookup    → embedding table
              │                           compute_numeric_stats     → numeric_stats.json (legacy_zscore only)
              │                           pretokenize_embedding
              │                                       │
   train_ehr_fm (--input_mode             train_ehr_fm --input_mode embedding
        discrete, default)                  --embedding_lookup_path … [--use_numerical_path]
              └─────────────────┬───────────────────┘
                                │
                     compute_representations         (match the training --input_mode)
```

### Shared steps

1. **Preprocessing** (`apply_meds_transforms`) — optional. Fix data-leakage issues such as events timestamped at admission instead of discharge (in MIMIC/OMOP-derived data).
2. **Vocabulary** (`train_vocabulary`) — build a token vocabulary (`vocab.json`) from the dataset. Supports joint and factorized tokenization modes.

### Discrete mode

3. **Tokenization** (`pretokenize`) — convert events to integer token sequences using `vocab.json`.
4. **Training** (`train_ehr_fm`) — train on tokenized data (`--input_mode discrete`, the default).

### Embedding mode

3. **Embedding lookup** (`build_embedding_lookup`) — encode each unique `embedding_text` string with a frozen text model into a lookup table (run once per dataset).
4. **Numeric stats** (`compute_numeric_stats`) — optional; precompute per-code log statistics (`numeric_stats.json`) needed only by the `legacy_zscore` numeric pathway.
5. **Tokenization** (`pretokenize_embedding`) — convert events to event-level rows of (embedding-text id, next-token label, numeric feature vector).
6. **Training** (`train_ehr_fm --input_mode embedding --embedding_lookup_path …`) — add `--use_numerical_path` for the dual-path (FiLM numerical) encoder, or `--no-use_numerical_path` for text-only.

### Both modes

7. **Representations** (`compute_representations`) — extract patient embeddings from a trained model, using the same `--input_mode` it was trained with.

### Model / input-mode matrix

| Model variant     | Input mode | Needs embedding lookup | Transfer across datasets |
| ----------------- | ---------- | ---------------------- | ------------------------ |
| `*_joint_ntp`     | discrete   | No                     | No (vocabulary-specific) |
| `*_emb_only_ntp`  | embedding  | Yes                    | Yes                      |
| `*_dual_path_ntp` | embedding  | Yes (+ numerical path) | Yes                      |

Only embedding-mode models transfer across datasets; discrete models are tied to their training vocabulary.

## Further Reading

- [Tutorial](tutorials/tutorial.ipynb) — end-to-end notebook walking through vocabulary training, tokenization, model training, and representation extraction on synthetic data
- [CLI Documentation](ehr_fm/scripts/README.md) — detailed usage, arguments, and examples for all eight commands
- [Data Module](ehr_fm/data/README.md) — dataset classes, collation, and batch sampling
