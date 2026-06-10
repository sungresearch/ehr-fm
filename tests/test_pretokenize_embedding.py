"""Tests for the embedding-mode pretokenize worker (_init_worker + _process_row).

Exercises the core per-row logic: embedding_text -> id mapping, OOV handling,
NTP-label emission (including OOV/cutoff masking), numeric features per pathway
mode, and age computation.
"""

import datetime
import json

import numpy as np
from safetensors.numpy import save_file

import ehr_fm.pretokenize.embedding_worker as pe

T0 = datetime.datetime(2000, 1, 1)


def _setup_worker(
    tmp_path, *, vocab_codes, id_mapping, vocab_size, numeric_pathway_mode="ref_range_priority"
):
    """Write a tiny vocab.json + EmbeddingLookup and initialize the worker state."""
    vocab = {
        "vocab": [{"type": "code", "code_string": c} for c in vocab_codes],
        "age_stats": {"mean": 0.0, "std": 86400.0},  # std = 1 day in seconds
        "config": {
            "emit_quantiles": False,
            "emit_text": False,
            "emit_stage": False,
            "num_quantiles": 10,
            "remove_prefixes": False,
            "separator": "/",
        },
        "quantile_breaks": {},
        "discovered_stages": [],
    }
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))

    look = tmp_path / "lookup"
    look.mkdir()
    (look / "id_mapping.json").write_text(json.dumps(id_mapping))
    save_file(
        {"embedding_table": np.zeros((len(id_mapping), 4), dtype=np.float16)},
        str(look / "embedding_table.safetensors"),
    )
    (look / "metadata.json").write_text(json.dumps({"model_name": "test"}))

    pe._init_worker(
        str(tmp_path / "vocab.json"), str(look), None, vocab_size, numeric_pathway_mode=numeric_pathway_mode
    )


def _event(code, embedding_text, day, **extra):
    return {
        "time": T0 + datetime.timedelta(days=day),
        "code": code,
        "embedding_text": embedding_text,
        **extra,
    }


def _row(seq, subject_id=7):
    return ((subject_id, T0 + datetime.timedelta(days=99)), seq)


class TestProcessRow:
    def test_maps_embedding_text_labels_and_features(self, tmp_path):
        _setup_worker(
            tmp_path,
            vocab_codes=["MEDS_BIRTH", "LAB/glucose"],
            id_mapping={"birth": 0, "glucose": 1},
            vocab_size=2,
        )
        seq = [
            _event("MEDS_BIRTH", "birth", 0),
            _event("LAB/glucose", "glucose", 1, numeric_value=5.0, ref_low=4.0, ref_high=6.0),
        ]
        out = pe._process_row(_row(seq), vocab_size=2)

        assert out["subject_id"] == 7
        assert out["length"] == 2
        assert out["embedding_text_ids"].to_pylist() == [0, 1]
        assert out["token_ids"].to_pylist() == [0, 1]
        assert out["age"].to_pylist() == [0.0, 1.0]  # days since MEDS_BIRTH
        # ref_range_priority: glucose 5 in [4, 6] -> midpoint 0.5, [x, is_refrange, is_log1p, present].
        assert out["numeric_features"][1].to_pylist() == [0.5, 1.0, 0.0, 1.0]

    def test_oov_embedding_text_event_dropped(self, tmp_path):
        _setup_worker(tmp_path, vocab_codes=["MEDS_BIRTH"], id_mapping={"birth": 0}, vocab_size=1)
        seq = [_event("MEDS_BIRTH", "birth", 0), _event("LAB/x", "NOT_IN_MAPPING", 1)]
        out = pe._process_row(_row(seq), vocab_size=1)
        assert out["length"] == 1
        assert out["embedding_text_ids"].to_pylist() == [0]

    def test_oov_code_label_is_masked(self, tmp_path):
        # Embedding text maps fine, but the code is absent from vocab -> NTP label -100.
        _setup_worker(tmp_path, vocab_codes=["MEDS_BIRTH"], id_mapping={"birth": 0, "g": 1}, vocab_size=1)
        seq = [_event("MEDS_BIRTH", "birth", 0), _event("LAB/not_in_vocab", "g", 1)]
        out = pe._process_row(_row(seq), vocab_size=1)
        assert out["embedding_text_ids"].to_pylist() == [0, 1]
        assert out["token_ids"].to_pylist() == [0, -100]

    def test_vocab_size_cutoff_masks_label(self, tmp_path):
        # 3 codes in vocab but vocab_size=2 -> the 3rd code's id (2) >= 2 -> -100.
        _setup_worker(
            tmp_path,
            vocab_codes=["MEDS_BIRTH", "A", "B"],
            id_mapping={"birth": 0, "a": 1, "b": 2},
            vocab_size=2,
        )
        seq = [_event("MEDS_BIRTH", "birth", 0), _event("A", "a", 1), _event("B", "b", 2)]
        out = pe._process_row(_row(seq), vocab_size=2)
        assert out["token_ids"].to_pylist() == [0, 1, -100]

    def test_legacy_zscore_produces_5dim_features(self, tmp_path):
        _setup_worker(
            tmp_path,
            vocab_codes=["MEDS_BIRTH", "LAB/glucose"],
            id_mapping={"birth": 0, "glucose": 1},
            vocab_size=2,
            numeric_pathway_mode="legacy_zscore",
        )
        seq = [
            _event("MEDS_BIRTH", "birth", 0),
            _event("LAB/glucose", "glucose", 1, numeric_value=5.0, ref_low=4.0, ref_high=6.0),
        ]
        out = pe._process_row(_row(seq), vocab_size=2)
        assert len(out["numeric_features"][1].to_pylist()) == 5

    def test_all_events_oov_returns_none(self, tmp_path):
        _setup_worker(tmp_path, vocab_codes=["MEDS_BIRTH"], id_mapping={"birth": 0}, vocab_size=1)
        seq = [_event("LAB/x", "NOPE", 0)]  # embedding_text not in mapping -> no usable events
        assert pe._process_row(_row(seq), vocab_size=1) is None
