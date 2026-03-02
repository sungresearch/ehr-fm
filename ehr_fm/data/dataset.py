import random
from datetime import datetime
from pathlib import Path
from typing import Any

import meds_reader
import polars as pl
import torch
from torch.utils.data import Dataset

from ehr_fm.logger import setup_logging
from ehr_fm.types import ConfigLike, PathLike
from ehr_fm.validation import MEDSReaderDatasetConfig, validate_config


def _get_event_attribute(event, attr_name, default=None):
    """Get an optional attribute from a MEDS reader event, returning *default* if absent."""
    return getattr(event, attr_name, default)


def _event_to_dict(event) -> dict:
    """Convert a MEDS reader event object to a plain dict."""
    d = {
        "code": event.code,
        "description": event.description,
        "numeric_value": _get_event_attribute(event, "numeric_value"),
        "text_value": _get_event_attribute(event, "text_value"),
        "unit": _get_event_attribute(event, "unit"),
        "time": event.time,
    }
    for optional in ("visit_id", "workflow_stage"):
        val = _get_event_attribute(event, optional)
        if val is not None:
            d[optional] = val
    return d


def _is_meds_reader_format(dataset_path: Path) -> bool:
    """Check if dataset_path contains meds_reader format data."""
    required_dirs = ["code", "description", "numeric_value", "time"]
    required_files = ["subject_id", "meds_reader.length", "meds_reader.properties", "meds_reader.version"]

    return (
        dataset_path.is_dir()
        and all((dataset_path / dir_name).is_dir() for dir_name in required_dirs)
        and all((dataset_path / file_name).exists() for file_name in required_files)
    )


class MEDSReaderDataset(Dataset):
    def __init__(self, config: ConfigLike):
        self.config = validate_config(config, MEDSReaderDatasetConfig)
        self.meds_reader_path = self.config.meds_reader_path
        self.samples_path = self.config.samples_path
        self.meds_db = meds_reader.SubjectDatabase(str(self.meds_reader_path), num_threads=1)
        self.split = self.config.split
        self.transform = self.config.transform
        self.logger = setup_logging(child_name=self.__class__.__name__)

        # Load sample dataframe
        cols_to_select = ["id", "index_t"]
        query = pl.scan_parquet(self.samples_path)

        if self.split is not None:
            cols_to_select.append("split")
            query = query.filter(pl.col("split") == self.split)

        query = query.select(cols_to_select).with_columns(pl.col("id").cast(pl.Int64))
        self.sample_df = query.collect()

        self.logger.info(f"MEDS data path (n={len(self.meds_db)}): {self.meds_reader_path}")
        self.logger.info(f"Samples path (n={len(self)}): {self.samples_path}")

    def __len__(self) -> int:
        return len(self.sample_df)

    def __getitem__(self, idx: int) -> tuple[tuple[int, datetime], Any] | None:
        sid = self.sample_df["id"][idx]
        index_t = self.sample_df["index_t"][idx]
        try:
            events = self.meds_db[sid].events
        except KeyError:
            self.logger.warning(f"Subject ID {sid} not found in MEDS database. Skipping index {idx}.")
            return None

        sample = []

        # Birth event is always prepended regardless of index_t ordering
        birth_event = next((e for e in events if e.code == "MEDS_BIRTH"), None)
        if birth_event is None:
            self.logger.warning(f"Subject ID {sid} has no MEDS_BIRTH event. Skipping index {idx}.")
            return None

        sample.append(_event_to_dict(birth_event))

        for event in events:
            if event.time > index_t:
                break
            if event.code == "MEDS_BIRTH":
                continue
            sample.append(_event_to_dict(event))

        if self.transform:
            sample = self.transform(sample)
        return ((sid, index_t), sample)


class TokenizedDataset(Dataset):
    """
    A PyTorch-style Dataset that reads a Parquet file of tokenized patient records
    and produces variable-length subsequences (windows) of token IDs for language modeling.

    Each record in the input Parquet must contain at least the following columns:
      - subject_id:        Unique patient identifier
      - index_time:        Timestamp of the index time (as datetime)
      - token_ids:         List of integer token IDs for the patient's event sequence
      - age:               List of patient ages at each token position
      - age_normalized:    List of normalized ages at each token position
      - length:            Integer length of the token_ids list

    Parameters
    ----------
    parquet_path : PathLike
        Filesystem path to the input Parquet file.
    max_length : int, optional (default=2048)
        Maximum allowed number of tokens in each window. Actual window lengths
        will be <= `max_length` (they'll be shorter if a patient's total sequence
        is shorter than this, or when sliding windows from the end).
    stride : int or None, optional (default=None)
        Step size for sliding windows when `one_window=False`. If None, defaults
        to `max_length` (i.e., non-overlapping windows).
    one_window : bool, optional (default=True)
        If True, generate exactly one window per patient ending at the last token.
        If False, generate multiple end-anchored windows by stepping backwards
        from the end in increments of `stride`.
    dropout_prob : float, optional (default=0.0)
        Probability of dropping out a token in the token_ids list.
        Note that combining this with TokenBudgetBatchSampler will result in less than
        `max_length` tokens in the output windows.
    min_length : int, optional (default=-1)
        Minimum allowed number of tokens in each window.

    Attributes
    ----------
    table : polars.DataFrame
        In-memory table loaded from the Parquet, containing all patient rows.
    window_index : List[Tuple[int, int]]
        List of `(row_idx, start_offset)` pairs indicating which slice to return
        for each window.


    Example
    -------
    >>> # Single end-anchored window per patient
    >>> ds = TokenizedDataset("data/patients.parquet", max_length=1024, one_window=True)
    >>> print(len(ds))                # number of patients
    >>> sample = ds[0]
    >>> print(sample["input_ids"].shape)
    torch.Size([window_length])

    >>> # Multiple sliding windows per patient
    >>> ds2 = TokenizedDataset("data/patients.parquet", max_length=1024, stride=512, one_window=False)
    >>> print(len(ds2))               # > number of patients
    >>> sample = ds2[5]
    >>> print(sample["offset"])       # e.g. might be 512, then 0
    """

    def __init__(
        self,
        parquet_path: PathLike,
        max_length: int = 2048,
        stride: int = None,
        one_window: bool = True,
        dropout_prob: float = 0.0,
        min_length: int = -1,
        num_ntp_classes: int = 8_192,
        convert_ages_to_positions: bool = False,
    ):
        self.max_length = max_length
        self.stride = stride or max_length
        self.one_window = one_window
        self.dropout_prob = dropout_prob
        self.min_length = min_length
        self.num_ntp_classes = num_ntp_classes
        self.convert_ages_to_positions = convert_ages_to_positions
        self.table = pl.read_parquet(
            parquet_path,
            memory_map=True,
            use_pyarrow=True,
        )

        self.window_index = []
        lengths = self.table["length"].to_list()

        for row_idx, length in enumerate(lengths):
            # Skip patients with fewer than min_length total tokens
            if self.min_length > 0 and length < self.min_length:
                continue

            # compute the "last-window" start because we start from the end of the sequence (more recent events)
            last_start = max(0, length - self.max_length)

            if self.one_window:
                # Calculate window length for the single window
                window_len = length - last_start  # Equivalent to min(length, max_length)
                # Skip if this single window is shorter than min_length
                if self.min_length > 0 and window_len < self.min_length:
                    continue
                # Add the single window
                self.window_index.append((row_idx, last_start))
            else:
                # sliding windows stepping backwards from the end
                # e.g. last_start, last_start - stride, last_start - 2*stride, ... down to >=0
                start = last_start
                while True:
                    # Calculate window length for the current window
                    end = min(start + self.max_length, length)
                    window_len = end - start

                    # Add window only if its length meets the minimum requirement
                    if self.min_length <= 0 or window_len >= self.min_length:
                        self.window_index.append((row_idx, start))

                    # Exit condition: If we just processed the window starting at 0
                    if start == 0:
                        break

                    # Move to the next window start
                    start = max(0, start - self.stride)

                    # Avoid adding duplicate window starting at 0 if stride doesn't align
                    if start == 0 and (row_idx, 0) in [(r, s) for r, s in self.window_index if r == row_idx]:
                        break

    def __len__(self):
        return len(self.window_index)

    # One patient & index_time combination
    def _get_row(self, row_idx):
        tbl = self.table[row_idx]

        return (
            tbl["subject_id"][0],
            tbl["index_time"][0],
            tbl["token_ids"][0].to_list(),
            tbl["age"][0].to_list(),
            tbl["age_normalized"][0].to_list(),
        )

    # One window
    def __getitem__(self, i):
        row_idx, start = self.window_index[i]
        pid, idx_t, toks, ages, ages_norm = self._get_row(row_idx)
        end = min(start + self.max_length, len(toks))

        slice_tok = toks[start:end]
        slice_age = ages[start:end]
        slice_ages_norm = ages_norm[start:end]

        # Apply dropout to the slice if needed
        if self.dropout_prob > 0.0 and len(slice_tok) > 0:
            keep_prob = 1.0 - self.dropout_prob
            mask = [random.random() < keep_prob for _ in range(len(slice_tok))]

            slice_tok = [tok for tok, keep in zip(slice_tok, mask) if keep]
            slice_age = [age for age, keep in zip(slice_age, mask) if keep]
            slice_ages_norm = [age_norm for age_norm, keep in zip(slice_ages_norm, mask) if keep]

            # Handle the edge case where dropout removes all tokens
            if not slice_tok:
                # Return the original slice before dropout (this is less ideal than returning None and
                # letting the caller handle it, but for now it's easier to implement, and better than
                # potentially crashing)
                slice_tok = toks[start:end]
                slice_age = ages[start:end]
                slice_ages_norm = ages_norm[start:end]

        # Convert potentially modified slices to tensors
        x = torch.as_tensor(slice_tok, dtype=torch.long)
        y = x.clone()
        y[:-1] = x[1:]

        if self.num_ntp_classes is not None:
            y[y >= self.num_ntp_classes] = -100

        y[-1] = -100

        # Apply age-to-position conversion if requested
        if self.convert_ages_to_positions:
            slice_age = [float(i) for i in range(len(slice_tok))]
            slice_ages_norm = [0.0] * len(slice_tok)

        return {
            "input_ids": x,
            "labels": y,
            "patient_id": pid,
            "index_time": idx_t.timestamp(),
            "offset": start,
            "length": x.size(0),
            "age": torch.as_tensor(slice_age, dtype=torch.float32),
            "age_normalized": torch.as_tensor(slice_ages_norm, dtype=torch.float32),
        }


def create_dataset(config: ConfigLike) -> MEDSReaderDataset:
    """Create a ``MEDSReaderDataset`` from a dict or config object.

    ``config`` must contain ``dataset_path`` pointing to a MEDS Reader directory.
    """
    if isinstance(config, dict):
        if "dataset_path" not in config:
            raise ValueError("Configuration must contain 'dataset_path'")
        dataset_path = Path(config["dataset_path"])
    elif hasattr(config, "dataset_path"):
        dataset_path = Path(config.dataset_path)
    else:
        raise ValueError(
            f"Config must have 'dataset_path'. "
            f"Got: {list(vars(config).keys()) if hasattr(config, '__dict__') else type(config)}"
        )

    if not _is_meds_reader_format(dataset_path):
        raise ValueError(
            f"Dataset path {dataset_path} is not in MEDS Reader format. "
            f"Expected directories (code/, description/, numeric_value/, time/) "
            f"and files (subject_id, meds_reader.length, etc.)."
        )

    if isinstance(config, dict):
        meds_reader_config = {**config, "meds_reader_path": config.pop("dataset_path")}
    else:
        meds_reader_config = type(config)(
            **{
                **{k: v for k, v in vars(config).items() if k != "dataset_path"},
                "meds_reader_path": config.dataset_path,
            }
        )
    return MEDSReaderDataset(meds_reader_config)
