"""Multi-worker equivalence + hang-safety test for the pretokenize pipeline.

The rest of the suite only ever exercises ``pretokenize_data`` with
``num_workers`` of 0 or 1 (the sequential path). The multiprocessing ``Pool``
path -- where the worker globals are set by the pool initializer in each child
process and read by the per-row task -- is never covered. That is exactly the
contract most at risk when the module is reorganized, and a broken contract
typically manifests as a *hang* (workers die in the initializer, the main
process blocks forever on the result queue).

This test:
  1. runs the parallel path (num_workers=2) in a fresh-interpreter subprocess
     with a hard timeout -- mirroring how production invokes the pool and
     avoiding fork-from-threaded-pytest artifacts -- so a real deadlock fails
     loudly instead of hanging the suite;
  2. asserts the parallel output is byte-identical to the sequential reference
     (sorted by subject, since the pool returns results unordered).
"""

import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import polars as pl
import pytest

from ehr_fm.io import write_dict_to_json
from ehr_fm.pretokenize import pretokenize_data

# The pretokenize import path used by the parallel subprocess. Kept as a module
# constant so the split (ehr_fm.tokenizer -> ehr_fm.pretokenize) is a one-line edit.
_PRETOKENIZE_IMPORT = "from ehr_fm.pretokenize import pretokenize_data"

# Generous relative to the tiny fixture; only trips on a real deadlock.
_WATCHDOG_TIMEOUT_S = 120


def _run_parallel_in_fresh_interpreter(*, vocab_path, out_dir, dataset_path, samples_path, num_workers):
    """Run the parallel pretokenize in a brand-new interpreter, like production.

    Using a fresh ``python -c`` subprocess (rather than forking the multi-threaded
    pytest process) mirrors how ``scripts/pretokenize.py`` invokes the pool and
    avoids fork-from-threaded-parent deadlocks that are pure test artifacts. The
    subprocess timeout converts a genuine pool deadlock into a loud failure
    instead of a hung suite.
    """
    code = textwrap.dedent(
        f"""
        {_PRETOKENIZE_IMPORT}
        pretokenize_data(
            vocab_path={str(vocab_path)!r},
            out_dir={str(out_dir)!r},
            dataset_path={str(dataset_path)!r},
            samples_path={str(samples_path)!r},
            split="train",
            num_workers={int(num_workers)},
        )
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            timeout=_WATCHDOG_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"pretokenize_data(num_workers={num_workers}) did not finish within "
            f"{_WATCHDOG_TIMEOUT_S}s -- likely a multiprocessing deadlock."
        )
    assert proc.returncode == 0, (
        f"parallel pretokenize subprocess failed (rc={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


@pytest.fixture
def multi_patient_dataset(tmp_path, convert_meds_to_reader):
    """A MEDS Reader dataset with enough patients to spread across workers."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    birth_time = datetime(2000, 1, 1)
    records = []
    n_patients = 12
    for pid in range(1, n_patients + 1):
        records.append(
            {
                "subject_id": pid,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            }
        )
        # Vary glucose values per patient so a cross-worker state leak would
        # surface as a wrong quantile token.
        n_labs = (pid % 3) + 1
        for j in range(n_labs):
            records.append(
                {
                    "subject_id": pid,
                    "time": birth_time + timedelta(days=30 * (j + 1)),
                    "code": "LAB/glucose",
                    "numeric_value": float(60 + 20 * pid + 15 * j),
                    "text_value": None,
                    "description": "",
                    "unit": None,
                }
            )
        records.append(
            {
                "subject_id": pid,
                "time": birth_time + timedelta(days=200),
                "code": "DIAGNOSIS/diabetes",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            }
        )

    pl.DataFrame(records).with_columns([pl.col("unit").cast(pl.Int32)]).write_parquet(
        data_dir / "data_0.parquet"
    )

    pl.DataFrame(
        {
            "id": list(range(1, n_patients + 1)),
            "index_t": [birth_time + timedelta(days=300)] * n_patients,
            "split": ["train"] * n_patients,
        }
    ).write_parquet(metadata_dir / "samples.parquet")

    return convert_meds_to_reader(tmp_path)


@pytest.fixture
def factorized_vocab_path(tmp_path):
    """Factorized-mode vocab matching the multi-patient fixture's codes."""
    vocab_data = {
        "vocab": [
            {"type": "quantile", "label": "Q:1", "weight": -1.0},
            {"type": "quantile", "label": "Q:2", "weight": -1.0},
            {"type": "quantile", "label": "Q:3", "weight": -1.0},
            {"type": "quantile", "label": "Q:UNK", "weight": -1.0},
            {"type": "stage", "label": "STAGE:taken", "weight": -1.0},
            {"type": "stage", "label": "STAGE:UNK", "weight": -1.0},
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "glucose", "weight": -0.4},
            {"type": "code", "code_string": "diabetes", "weight": -0.3},
        ],
        "age_stats": {"mean": 2592000.0, "std": 1296000.0},
        "config": {
            "tokenization_mode": "factorized",
            "emit_quantiles": True,
            "emit_text": False,
            "emit_stage": False,
            "num_quantiles": 10,
            "remove_prefixes": True,
            "separator": "/",
        },
        "quantile_breaks": {"LAB/glucose": [100.0, 150.0]},
        "discovered_stages": ["taken"],
    }
    vocab_path = tmp_path / "factorized_vocab.json"
    write_dict_to_json(vocab_data, vocab_path, overwrite=True)
    return vocab_path


def _read_sorted(parquet_path):
    df = pl.read_parquet(parquet_path)
    return df.sort("subject_id")


class TestPretokenizeMultiWorker:
    def test_parallel_matches_sequential(self, multi_patient_dataset, factorized_vocab_path):
        """num_workers=2 (Pool) yields the same tokens as num_workers=0 (sequential)."""
        base_kwargs = dict(
            vocab_path=factorized_vocab_path,
            dataset_path=multi_patient_dataset,
            samples_path=multi_patient_dataset / "metadata" / "samples.parquet",
            split="train",
        )

        seq_dir = multi_patient_dataset / "pretok_seq"
        par_dir = multi_patient_dataset / "pretok_par"

        # Sequential reference (in-process; cannot deadlock).
        pretokenize_data(out_dir=seq_dir, num_workers=0, **base_kwargs)
        # Parallel path in a fresh interpreter, under a hard timeout watchdog.
        _run_parallel_in_fresh_interpreter(
            vocab_path=factorized_vocab_path,
            out_dir=par_dir,
            dataset_path=multi_patient_dataset,
            samples_path=multi_patient_dataset / "metadata" / "samples.parquet",
            num_workers=2,
        )

        seq = _read_sorted(seq_dir / "patients_tokenized.parquet")
        par = _read_sorted(par_dir / "patients_tokenized.parquet")

        assert seq.shape == par.shape
        assert seq["subject_id"].to_list() == par["subject_id"].to_list()
        for col in ("token_ids", "age", "age_normalized", "length"):
            assert seq[col].to_list() == par[col].to_list(), f"mismatch in column {col}"
