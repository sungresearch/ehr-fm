# EHR-FM Scripts

This directory contains command-line scripts for the EHR-FM training pipeline. Most steps (`train_vocabulary`, `pretokenize`, `compute_representations`) require data in **MEDS Reader** format. If your data is in standard MEDS format, see the [main README](../../README.md#data-format) for conversion instructions. The `apply_meds_transforms` step is the exception -- it operates on standard MEDS parquet files and can convert its output to MEDS Reader.

All scripts are exposed as CLI commands via the `pyproject.toml` configuration.

> **New to EHR-FM?** The [tutorial notebook](../../tutorials/tutorial.ipynb) walks through the full pipeline on synthetic data -- from vocabulary training to patient representations -- and is the fastest way to get started.

## Available Commands

- [`train_vocabulary`](#train_vocabulary) - Train vocabulary from a MEDS Reader dataset
- [`pretokenize`](#pretokenize) - Pretokenize a MEDS Reader dataset using a trained vocabulary
- [`train_ehr_fm`](#train_ehr_fm) - Train an EHR-FM model
- [`compute_representations`](#compute_representations) - Compute patient representations using a trained EHR-FM model
- [`apply_meds_transforms`](#apply_meds_transforms) - Apply transformations to a standard MEDS dataset

______________________________________________________________________

## `train_vocabulary`

Train a vocabulary from a MEDS Reader dataset using reservoir sampling for numeric values. Supports both **joint** and **factorized** tokenization modes.

### Input

**MEDS Reader** directory (`--dataset_path`). See the [main README](../../README.md#data-format) for conversion instructions if your data is in standard MEDS format.

Also reads a `samples.parquet` file (default: `{dataset_path}/metadata/samples.parquet`) with columns `id` (int64) and `index_t` (timestamp), plus an optional `split` column for filtering.

### Output

- `{out_dir}/vocab.json` -- token vocabulary, age statistics, quantile breaks, tokenization config, and discovered stages
- `{out_dir}/reservoirs.json` -- reservoir sampling data (only if `--save_reservoirs`)

### Required Arguments

- `--dataset_path` - Path to the MEDS Reader dataset
- `--out_dir` - Output directory for vocabulary files

### Common Options

- `--vocab_size` - Maximum vocabulary size (default: 98,304)
- `--split` - Data split to use (train, val, test, validation)
- `--samples_path` - Path to samples.parquet file
- `--numeric_sample_reservoir_size` - Size of reservoir for numeric value sampling (default: 1,000)
- `--seed` - Random seed for reproducibility (default: 444)
- `--save_reservoirs` - Save reservoir sampling data
- `--overwrite` - Overwrite existing vocabulary files

### Tokenization Mode Options

- `--tokenization_mode` - Vocabulary mode: `joint` (default) or `factorized`
  - **joint**: 1 token per event including base event code and attributes (e.g., `LAB/glucose|Q3` as a single token)
  - **factorized**: Separate base event code and attribute tokens (e.g., `LAB/glucose` + `Q:3` as two separate tokens)

### Attribute Token Options

These options apply to **both** `joint` and `factorized` tokenization modes:

- `--num_quantiles` - Number of quantile buckets for numeric values (default: 10)
- `--emit_quantiles` / `--no_emit_quantiles` - Include quantile tokens for numeric values (default: True)
- `--emit_text` / `--no_emit_text` - Include text value tokens (default: True)
- `--emit_stage` / `--no_emit_stage` - Include stage tokens discovered from data (default: True)
- `--remove_prefixes` / `--no_remove_prefixes` - Remove code prefixes in vocabulary (default: True)
- `--code_separator` - Code separator for prefix removal (default: `/`)

### Interval/Demographic Token Options

- `--add_time_interval_tokens` - Prepend time interval tokens to vocabulary
- `--add_demographic_tokens` - Prepend demographic tokens (sex, age buckets, year buckets)

### Unit-Specific Binning Options

- `--numeric_bin_by_unit` - Enable unit-specific binning for numeric values
- `--min_samples_per_unit` - Minimum samples required per (code, unit) pair

### Example Usage

```bash
# Basic joint vocabulary training
train_vocabulary \
  --dataset_path /path/to/meds_reader_data \
  --out_dir /path/to/vocab/output \
  --split train

# Factorized vocabulary
train_vocabulary \
  --dataset_path /path/to/meds_reader_data \
  --out_dir /path/to/vocab/output \
  --split train \
  --tokenization_mode factorized

# Factorized vocabulary with interval and demographic tokens
train_vocabulary \
  --dataset_path /path/to/meds_reader_data \
  --out_dir /path/to/vocab/output \
  --split train \
  --tokenization_mode factorized \
  --add_time_interval_tokens \
  --add_demographic_tokens
```

______________________________________________________________________

## `pretokenize`

Convert a MEDS Reader dataset into integer token sequences for model training. The tokenization mode (joint or factorized) is auto-detected from the `tokenization_mode` field in `vocab.json`.

**Event-level tokenization.** Each MEDS event is mapped to one or more token IDs via a tokenization policy (defined in `ehr_fm.tokenization`, selected automatically):

- **Joint mode** -- each event emits a single combined token (e.g., `LAB/glucose/Q:3`). The token string is looked up in the vocabulary; if not found, the event is out-of-vocabulary (OOV) and dropped.
- **Factorized mode** -- each event emits a base code token plus zero or more attribute tokens (quantile, text, stage). A no-orphans invariant is enforced: if the base code token is OOV, *all* attribute tokens for that event are dropped. Attribute tokens that are individually OOV are silently skipped without affecting the base or other attributes.

**Sequence assembly.** After event-level tokenization, the pipeline assembles each patient's token sequence by optionally injecting temporal interval tokens between events, prepending demographic prefix tokens, computing per-token age arrays, and filtering out any token with id >= `vocab_size` (vocab cutoff). The result is written to Parquet.

### Input

- **MEDS Reader** directory (`--dataset_path`). See the [main README](../../README.md#data-format) for conversion instructions.
- `vocab.json` from `train_vocabulary` (`--vocab_path`)
- `samples.parquet` (default: `{dataset_path}/metadata/samples.parquet`)

### Output

`{out_dir}/{output_filename}` (default `patients_tokenized.parquet`) with the following schema:

| Column           | Type            | Description                             |
| ---------------- | --------------- | --------------------------------------- |
| `subject_id`     | int64           | Patient identifier                      |
| `index_time`     | timestamp\[ns\] | Index time from samples file            |
| `token_ids`      | list\<int32>    | Token ID sequence                       |
| `age`            | list\<float32>  | Age in days at each token               |
| `length`         | int32           | Number of tokens                        |
| `age_normalized` | list\<float32>  | Z-scored age using vocab age statistics |

### Required Arguments

- `--dataset_path` - Path to the MEDS Reader dataset
- `--vocab_path` - Path to vocabulary file (`vocab.json` from `train_vocabulary`)
- `--out_dir` - Output directory for tokenized files

### Common Options

- `--split` - Data split to use (train, val, test, validation)
- `--samples_path` - Path to samples.parquet file
- `--workers` - Number of worker processes (0=single-process, -1=all cores, default: -1)
- `--max_events_per_patient` - Maximum events per patient (default: no limit)
- `--vocab_size` - Max vocab size (defaults to length of vocabulary)
- `--row_group_size` - Rows per group in output Parquet file (default: 32,768)
- `--output_filename` - Name of output Parquet file (default: "patients_tokenized.parquet")

### Time Interval Options

- `--inject_time_intervals` - Inject temporal interval tokens between events
- `--max_interval_repeat` - Maximum repeatable 6-month interval tokens (default: unlimited)

### Demographic Options

- `--demographic_prefix` - Prepend demographic tokens and start timeline at first event
- `--demographic_include_year` - Include year token in demographic prefix
- `--sex_codes_male` - Comma-separated code strings for male sex
- `--sex_codes_female` - Comma-separated code strings for female sex
- `--sex_codes_unknown` - Comma-separated code strings for unknown sex

### Example Usage

```bash
# Basic pretokenization (auto-detects mode from vocabulary)
pretokenize \
  --dataset_path /path/to/meds_reader_data \
  --vocab_path /path/to/vocab/output/vocab.json \
  --out_dir /path/to/tokenized/output \
  --split train

# Pretokenization with interval injection and demographics
pretokenize \
  --dataset_path /path/to/meds_reader_data \
  --vocab_path /path/to/vocab/output/vocab.json \
  --out_dir /path/to/tokenized/output \
  --split train \
  --inject_time_intervals \
  --demographic_prefix \
  --sex_codes_male 8507 \
  --sex_codes_female 8532

# Multi-worker pretokenization with custom settings
pretokenize \
  --dataset_path /path/to/meds_reader_data \
  --vocab_path /path/to/vocab/output/vocab.json \
  --out_dir /path/to/tokenized/output \
  --split train \
  --workers 8 \
  --max_events_per_patient 100000
```

______________________________________________________________________

## `train_ehr_fm`

Train an EHR-FM model using pretokenized data.

### Input

Pretokenized parquet files from `pretokenize` (one each for training and validation). The files must contain the columns documented in the `pretokenize` output schema above.

### Output

HuggingFace-style checkpoint directories under `{output_dir}/`:

- `checkpoint-{step}/` -- periodic checkpoints containing `config.json` and `model.safetensors`
- `checkpoint-topk-{step}/` -- best-K checkpoints (if `--save_top_k > 0`)
- `top_k_checkpoints.json` -- manifest of top-K checkpoints and their metrics (if `--save_top_k > 0`)

### Configuration

This command uses a **config file + CLI override** pattern:

1. A YAML/JSON config file is a **required positional argument**.
2. The config file is a flat dict whose keys map directly to CLI flag names (without `--`). For example, `n_layers: 22` in the config is equivalent to passing `--n_layers 22`.
3. CLI flags override any value set in the config file.
4. `train_ehr_fm --help` prints full usage and exits without requiring a config file.

Extraneous keys in the config file are silently ignored -- only the parameters listed below are forwarded to the model.

#### Boolean flags

Boolean flags (`--fp16`, `--bf16`, `--load_best_model_at_end`, `--greater_is_better`, `--save_best_checkpoint_only`, and their `_full_eval` variants) accept the following values:

- **From the CLI:** `True` / `False`, `yes` / `no`, `1` / `0` (case-insensitive). Example: `--bf16 True`.
- **From YAML config:** native YAML booleans (`true` / `false`). Example: `bf16: true`.

> **Note:** Do _not_ use bare `--bf16` without a value -- these are not on/off store-true flags. Always provide an explicit value.

#### `resume_from_checkpoint`

This flag controls checkpoint resumption and accepts three kinds of values:

| Value | Meaning |
|---|---|
| `null` (YAML) / omitted | Start training from scratch. |
| `true` (YAML) or `True` / `true` / `yes` / `1` (CLI) | **Auto-detect**: resume from the latest checkpoint found in `output_dir`. Regular checkpoints (`checkpoint-{step}`) are preferred; top-K checkpoints (`checkpoint-topk-{step}`) are used as a fallback. If no checkpoints exist, training starts from scratch. |
| A path string (e.g. `/path/to/checkpoint-5000`) | Resume from that specific checkpoint directory. |

### Required Arguments

- `config` (positional) - Path to YAML/JSON configuration file
- `--train_path` - Path to training data (pretokenized parquet)
- `--val_path` - Path to validation data (pretokenized parquet)
- `--output_dir` - Output directory for checkpoints and logs

### Key Configuration Options

All options below can be set in the config file or overridden via CLI flags.

#### Data Settings

- `--max_tokens_per_batch` - Maximum tokens per batch (default: 16,384)
- `--min_patients_per_batch` - Minimum patients per batch (default: 2)
- `--max_seq_length_per_patient` - Maximum sequence length per patient
- `--token_dropout_prob` - Token dropout probability (default: 0.0)
- `--min_patient_length` - Minimum patient sequence length (default: 2)
- `--stride` - Stride for sliding windows

#### Model Architecture

- `--vocab_size` - Vocabulary size (default: 98,304)
- `--hidden_size` - Hidden size (default: 768)
- `--n_layers` - Number of layers (default: 12)
- `--n_heads` - Number of attention heads (default: 12)
- `--attention_width` - Attention width (default: 496)
- `--intermediate_size` - FFN intermediate size (default: 1,152)
- `--use_normed_ages` - Use normalized ages (default: True)
- `--hidden_act` - Activation function (default: swiglu)

#### Other Model Options

- `--num_ntp_classes` - Number of next-token prediction classes (default: 8,192)
- `--convert_ages_to_positions` - Convert ages to sequential positions at runtime

#### Training Settings

- `--learning_rate` - Initial learning rate (default: 5e-5)
- `--max_steps` - Total training steps (default: 1,000,000)
- `--warmup_steps` - Linear warmup steps (default: 0)
- `--weight_decay` - Weight decay (default: 0.0)
- `--optimizer_name` - Optimizer: `adamw` or `stableadamw` (default: adamw)

#### Checkpointing & Early Stopping

- `--save_top_k` - Keep K best checkpoints based on metric (default: 0, disabled)
- `--metric_for_best_model` - Metric for best model selection (default: eval_loss)
- `--greater_is_better` - Higher metric is better (default: False)
- `--early_stopping_patience` - Evaluations without improvement before stopping (default: 0, disabled)
- `--early_stopping_threshold` - Minimum improvement threshold (default: 0.0)

### Example Usage

```bash
# Train with configuration file
train_ehr_fm config.yaml \
  --train_path /path/to/train_tokenized.parquet \
  --val_path /path/to/val_tokenized.parquet \
  --output_dir /path/to/model/output

# Train with CLI overrides
train_ehr_fm config.yaml \
  --train_path /path/to/train_tokenized.parquet \
  --val_path /path/to/val_tokenized.parquet \
  --output_dir /path/to/model/output \
  --learning_rate 1e-4 \
  --max_steps 500000 \
  --warmup_steps 10000
```

### Example Configuration File (config.yaml)

Keys map 1:1 to CLI flag names (without `--`). All parameters and their defaults are shown below.

```yaml
# --- Paths (typically passed via CLI, but can be set here) ---
# train_path: /path/to/train_tokenized.parquet
# val_path: /path/to/val_tokenized.parquet
# output_dir: /path/to/model/output
# initial_weights_path: null  # path to pretrained weights for warm-starting

# --- Model architecture ---
vocab_size: 98304
hidden_size: 768
intermediate_size: 1152
n_layers: 12
n_heads: 12
attention_width: 496
hidden_act: swiglu
use_normed_ages: true
use_bias: false
remove_first_block_norm: true
alternating_dense_layers: false
dense_every_n_layers: 3      # only used when alternating_dense_layers is true
rope_base_sparse: 100.0
rope_base_global: 10000.0
separate_rope_by_attention: false
num_ntp_classes: 8192

# --- Data ---
max_tokens_per_batch: 16384
min_patients_per_batch: 2
# max_seq_length_per_patient: null  # defaults to max_tokens_per_batch // min_patients_per_batch
min_patient_length: 2
token_dropout_prob: 0.0
# stride: null
# convert_ages_to_positions: false

# --- Optimizer ---
optimizer_name: adamw         # adamw | stableadamw
learning_rate: 5e-5
weight_decay: 0.0
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
max_grad_norm: 1.0

# --- LR schedule (HuggingFace built-in) ---
lr_scheduler_type: linear     # linear | cosine | constant | constant_with_warmup | cosine_with_min_lr
warmup_steps: 0

# --- LR schedule (custom, overrides lr_scheduler_type) ---
# lr_scheduler_name: wsd      # wsd (Warmup-Stable-Decay)
# wsd_num_decay_steps: null
# wsd_decay_type: cosine      # cosine | linear

# --- Training ---
max_steps: 1000000
gradient_accumulation_steps: 1
per_device_train_batch_size: 1  # keep at 1 with TokenBudgetBatchSampler
seed: 42

# --- Precision ---
fp16: false
bf16: false
# fp16_full_eval: false
# bf16_full_eval: false

# --- Logging & evaluation ---
logging_strategy: steps
logging_steps: 50
evaluation_strategy: steps
eval_steps: 50
# max_eval_batches: null      # limit eval batches (null = all)
report_to: all                # all | wandb | tensorboard | none

# --- Saving ---
save_strategy: steps
save_steps: 500
# save_total_limit: null      # max checkpoints to keep
# resume_from_checkpoint: null    # null = fresh start, true = auto-detect, or a path string

# --- Top-K checkpointing ---
save_top_k: 0                # 0 = disabled
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: true
# save_best_checkpoint_only: false
# skip_first_n_steps: 1000

# --- Variable save frequency (optional) ---
# early_save_until_step: 0
# early_save_every: null
# late_save_every: null

# --- Early stopping ---
early_stopping_patience: 0    # 0 = disabled
early_stopping_threshold: 0.0

# --- Weights & Biases ---
# run_name: null
# project_name: null
# tags: []

# --- DataLoader ---
dataloader_num_workers: 0
```

______________________________________________________________________

## `compute_representations`

Compute patient representations using a trained EHR-FM model. Accepts either a pretokenized dataset or a MEDS Reader directory (pretokenizes on-the-fly).

### Input (two modes, mutually exclusive)

- **Pretokenized** (`--pretokenized_dataset_path`): A parquet file from `pretokenize`.
- **MEDS Reader** (`--dataset_path`): A MEDS Reader directory + `vocab.json` (`--vocab_path`). The data is pretokenized into a temporary directory at runtime.

Both modes require `--model_path` pointing to a trained checkpoint from `train_ehr_fm`.

### Output

Parquet file at `--output_path` with the following schema:

| Column            | Type           | Description                                          |
| ----------------- | -------------- | ---------------------------------------------------- |
| `id`              | string         | Subject ID                                           |
| `index_t`         | datetime       | Index time                                           |
| `representations` | list\<float32> | Last-token hidden state (length = model hidden_size) |

### Required Arguments (mutually exclusive)

- `--dataset_path` - Path to MEDS Reader directory (will pretokenize on-the-fly)
- `--pretokenized_dataset_path` - Path to existing pretokenized parquet file (skips pretokenization)

### Other Required Arguments

- `--model_path` - Path to pretrained EHR-FM model
- `--output_path` - Path to save computed representations

### Common Options

- `--vocab_path` - Path to vocabulary file (required when using `--dataset_path`)
- `--split` - Data split to process (train, validation, test)
- `--samples_path` - Path to samples.parquet file
- `--num_workers` - Number of workers for pretokenization (-1=all cores)
- `--max_tokens_per_batch` - Maximum tokens per batch (default: 16,384)
- `--min_patients_per_batch` - Minimum patients per batch (default: 1)
- `--use_fp16` - Use FP16 precision
- `--use_bfloat16` - Use BFloat16 precision

### Other Options

- `--vocab_size` - Max vocab size (defaults to length of vocabulary)
- `--convert_ages_to_positions` - Convert ages to sequential positions at runtime

### Example Usage

```bash
# Using pretokenized data (recommended for large datasets)
compute_representations \
  --pretokenized_dataset_path /path/to/pretokenized.parquet \
  --model_path /path/to/trained/model \
  --vocab_path /path/to/vocab.json \
  --output_path /path/to/representations.parquet

# Using MEDS Reader data (will pretokenize on-the-fly)
compute_representations \
  --dataset_path /path/to/meds_reader_data \
  --model_path /path/to/trained/model \
  --vocab_path /path/to/vocab/output/vocab.json \
  --output_path /path/to/representations.parquet \
  --split validation

# With precision settings for memory efficiency
compute_representations \
  --pretokenized_dataset_path /path/to/pretokenized.parquet \
  --model_path /path/to/trained/model \
  --output_path /path/to/representations.parquet \
  --use_bfloat16 \
  --max_tokens_per_batch 32768
```

______________________________________________________________________

## `apply_meds_transforms`

Apply optional preprocessing transforms to a MEDS dataset. Currently the only
supported transform is the **visit end time fix**, which corrects data leakage
in OMOP-derived datasets (e.g. MIMIC) where diagnosis, procedure, and
observation events are timestamped at visit start instead of visit end.

**When to use this command:** Only when your MEDS data has visit-end-time
leakage. If your ETL already assigns correct event timestamps you can skip
this step entirely.

### Input

Standard **MEDS** parquet dataset. Expects parquet files under `data/` (or at the directory root) with required columns: `subject_id`, `time`, `code`. Optional columns for the visit-end-time transform: `workflow_stage`, `visit_id`, `description`. The `metadata/` directory, if present, is copied verbatim to the output.

### Output

- `{output_dir}/data/*.parquet` -- transformed MEDS parquet files
- `{output_dir}/metadata/` -- copied from input
- If `--convert-to-meds-reader` (default: enabled): a MEDS Reader database at `{output_dir}_meds_reader/` (override with `--meds-reader-output-dir`)

### Usage Modes

1. **Config file mode**: `apply_meds_transforms config.yaml`
2. **CLI arguments mode**: `apply_meds_transforms --input-dir ... --output-dir ...`
3. **Mixed mode**: `apply_meds_transforms config.yaml --input-dir ... --output-dir ...`

### Required Arguments

- `--input-dir` - Directory containing input MEDS data
- `--output-dir` - Directory to save transformed data

### Transform Options

- `--move-to-visit-end` - Move events from visit start to visit end time (fixes data leakage)

### Visit End Time Options

Uses the `workflow_stage` column to identify visit start/end events:

- `--visit-end-code-prefixes` - Code prefixes to move (e.g., DIAGNOSIS/ PROCEDURE/ OBSERVATION/)
- `--visit-end-check-description` / `--no-visit-end-check-description` - Also check description field (default: True)
- `--visit-code-pattern` - Pattern to identify visit events (default: VISIT)
- `--workflow-stage-start` - Value indicating visit start (default: start, case-insensitive)
- `--workflow-stage-end` - Value indicating visit end (default: end, case-insensitive)

### MEDS Reader Options

- `--convert-to-meds-reader` - Convert output to MEDS Reader format (default: True)
- `--no-convert-to-meds-reader` - Skip conversion
- `--meds-reader-output-dir` - Custom output directory for MEDS Reader
- `--verify-meds-reader` - Verify MEDS Reader dataset after conversion

### Dataset Metadata Options

- `--dataset-name` - Name of the dataset (default: "Transformed Dataset")
- `--dataset-version` - Version of the dataset (default: "1.0")

### Example Usage

```bash
# Apply visit end time fix
apply_meds_transforms \
  --input-dir /path/to/meds/data \
  --output-dir /path/to/transformed/output \
  --move-to-visit-end

# With custom visit configuration
apply_meds_transforms \
  --input-dir /path/to/meds/data \
  --output-dir /path/to/transformed/output \
  --move-to-visit-end \
  --visit-end-code-prefixes DIAGNOSIS/ PROCEDURE/ OBSERVATION/ \
  --visit-code-pattern VISIT

# Skip MEDS Reader conversion
apply_meds_transforms \
  --input-dir /path/to/meds/data \
  --output-dir /path/to/transformed/output \
  --move-to-visit-end \
  --no-convert-to-meds-reader
```

### Example Configuration File

```yaml
# Input and output directories
input_dir: /path/to/meds/data
output_dir: /path/to/transformed/output

# Transform configuration
transforms:
  move_to_visit_end: true

  visit_end_time_config:
    code_prefixes:
      - DIAGNOSIS/
      - PROCEDURE/
      - OBSERVATION/
    check_description: true
    visit_code_pattern: VISIT
    workflow_stage_start: start
    workflow_stage_end: end

  convert_to_meds_reader: true
  verify_meds_reader: true
```
