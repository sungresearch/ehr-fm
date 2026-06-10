"""Multi-worker equivalence + hang-safety test for the embedding-mode pretokenizer.

Mirrors test_pretokenize_multiworker.py for the embedding pipeline, whose Pool
path is otherwise untested (the existing test_pretokenize_embedding.py only drives
_init_worker/_process_row in-process). Runs the CLI in a fresh-interpreter
subprocess with a hard timeout -- so a genuine pool deadlock fails loudly instead
of hanging the suite -- and asserts the parallel output (--workers 2) is
byte-identical to the sequential output (--workers 0), for both numeric pathway
modes (their feature-construction code paths are independent).

Because the embedding pretokenizer currently exposes only an argparse ``main()``,
both paths are driven through the CLI. The comparison is invariant to the
refactor that moves the logic into ehr_fm.pretokenize.embedding_*: the CLI
contract is unchanged.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta

import polars as pl
import pytest

_WATCHDOG_TIMEOUT_S = 120


def _run_embedding_cli(
    *, dataset_path, vocab_path, lookup_dir, out_dir, num_workers, numeric_pathway_mode, numeric_stats_path
):
    """Run the embedding pretokenize CLI in a fresh interpreter, under a timeout."""
    cmd = [
        sys.executable,
        "-m",
        "ehr_fm.scripts.pretokenize_embedding",
        "--dataset_path",
        str(dataset_path),
        "--vocab_path",
        str(vocab_path),
        "--embedding_lookup_path",
        str(lookup_dir),
        "--samples_path",
        str(dataset_path / "metadata" / "samples.parquet"),
        "--out_dir",
        str(out_dir),
        "--split",
        "train",
        "--workers",
        str(num_workers),
        "--numeric_pathway_mode",
        numeric_pathway_mode,
    ]
    if numeric_stats_path is not None:
        cmd += ["--numeric_stats_path", str(numeric_stats_path)]
    try:
        proc = subprocess.run(cmd, timeout=_WATCHDOG_TIMEOUT_S, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"pretokenize_embedding(--workers {num_workers}) did not finish within "
            f"{_WATCHDOG_TIMEOUT_S}s -- likely a multiprocessing deadlock."
        )
    assert proc.returncode == 0, (
        f"embedding pretokenize subprocess failed (rc={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


@pytest.fixture
def embedding_dataset(tmp_path, convert_meds_to_reader):
    """MEDS Reader dataset with embedding_text + ref ranges, multiple patients."""
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
                "embedding_text": "birth",
                "ref_low": None,
                "ref_high": None,
                "description": "",
            }
        )
        n_labs = (pid % 3) + 1
        for j in range(n_labs):
            has_ref = j % 2 == 0  # exercise both ref-range and log1p/zscore branches
            records.append(
                {
                    "subject_id": pid,
                    "time": birth_time + timedelta(days=30 * (j + 1)),
                    "code": "LAB/glucose",
                    "numeric_value": float(3 + pid + 2 * j),
                    "embedding_text": "glucose",
                    "ref_low": 4.0 if has_ref else None,
                    "ref_high": 6.0 if has_ref else None,
                    "description": "",
                }
            )
        records.append(
            {
                "subject_id": pid,
                "time": birth_time + timedelta(days=150),
                "code": "DIAGNOSIS/diabetes",
                "numeric_value": None,
                "embedding_text": "diabetes dx",
                "ref_low": None,
                "ref_high": None,
                "description": "",
            }
        )

    df = pl.DataFrame(records)
    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))
    df.write_parquet(data_dir / "data_0.parquet")

    pl.DataFrame(
        {
            "id": list(range(1, n_patients + 1)),
            "index_t": [birth_time + timedelta(days=300)] * n_patients,
            "split": ["train"] * n_patients,
        }
    ).write_parquet(metadata_dir / "samples.parquet")

    return convert_meds_to_reader(tmp_path)


@pytest.fixture
def embedding_vocab_path(tmp_path):
    """Joint-mode vocab (base-code tokens) matching the fixture's codes."""
    vocab_data = {
        "vocab": [
            {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
            {"type": "code", "code_string": "glucose", "weight": -0.4},
            {"type": "code", "code_string": "diabetes", "weight": -0.3},
        ],
        "age_stats": {"mean": 2592000.0, "std": 1296000.0},
        "config": {
            "tokenization_mode": "joint",
            "emit_quantiles": False,
            "emit_text": False,
            "emit_stage": False,
            "num_quantiles": 10,
            "remove_prefixes": True,
            "separator": "/",
        },
        "quantile_breaks": {"LAB/glucose": [5.0, 8.0, 12.0]},
        "discovered_stages": [],
    }
    p = tmp_path / "embedding_vocab.json"
    p.write_text(json.dumps(vocab_data))
    return p


@pytest.fixture
def embedding_lookup_dir(tmp_path):
    """Embedding lookup dir holding just the id_mapping.json the worker reads."""
    look = tmp_path / "lookup"
    look.mkdir()
    (look / "id_mapping.json").write_text(json.dumps({"birth": 0, "glucose": 1, "diabetes dx": 2}))
    return look


@pytest.fixture
def numeric_stats_path(tmp_path):
    """numeric_stats.json keyed by full code, for the legacy_zscore pathway."""
    p = tmp_path / "numeric_stats.json"
    p.write_text(json.dumps({"LAB/glucose": {"log_mean": 2.0, "log_std": 0.5}}))
    return p


def _read_sorted(parquet_path):
    return pl.read_parquet(parquet_path).sort("subject_id")


class TestEmbeddingPretokenizeMultiWorker:
    @pytest.mark.parametrize("pathway_mode", ["ref_range_priority", "legacy_zscore"])
    def test_parallel_matches_sequential(
        self,
        embedding_dataset,
        embedding_vocab_path,
        embedding_lookup_dir,
        numeric_stats_path,
        pathway_mode,
    ):
        stats = numeric_stats_path if pathway_mode == "legacy_zscore" else None
        seq_dir = embedding_dataset / f"emb_seq_{pathway_mode}"
        par_dir = embedding_dataset / f"emb_par_{pathway_mode}"

        common = dict(
            dataset_path=embedding_dataset,
            vocab_path=embedding_vocab_path,
            lookup_dir=embedding_lookup_dir,
            numeric_pathway_mode=pathway_mode,
            numeric_stats_path=stats,
        )
        _run_embedding_cli(out_dir=seq_dir, num_workers=0, **common)
        _run_embedding_cli(out_dir=par_dir, num_workers=2, **common)

        seq = _read_sorted(seq_dir / "patients_tokenized.parquet")
        par = _read_sorted(par_dir / "patients_tokenized.parquet")

        assert seq.shape == par.shape
        assert seq["subject_id"].to_list() == par["subject_id"].to_list()
        for col in ("embedding_text_ids", "token_ids", "numeric_features", "age", "age_normalized", "length"):
            assert seq[col].to_list() == par[col].to_list(), f"mismatch in column {col}"
