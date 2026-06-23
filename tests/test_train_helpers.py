"""Tests for train_ehr_fm helpers: build_optimizer, prepare_data, prepare_model."""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.numpy import save_file

from ehr_fm.data import TokenBudgetBatchSampler, TokenizedDataset
from ehr_fm.logger import setup_logging
from ehr_fm.models import DualPathInputEncoder
from ehr_fm.models.config import EHRFMConfig
from ehr_fm.models.transformer import EHRFM
from ehr_fm.scripts.train_ehr_fm import build_optimizer, prepare_data, prepare_model

_DEVICE = torch.device("cpu")
_LOGGER = setup_logging()


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------
def _model():
    cfg = EHRFMConfig(
        transformer={
            "vocab_size": 50,
            "hidden_size": 16,
            "intermediate_size": 24,
            "n_heads": 2,
            "n_layers": 2,
            "attention_width": 8,
            "input_mode": "discrete",
        },
        task={"task_type": "sequence_classification", "n_classes": 50},
    )
    return EHRFM(cfg)


def test_build_optimizer_weight_decay_grouping():
    model = _model()
    opt = build_optimizer(
        config={"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
        model=model,
        logger=_LOGGER,
    )

    assert len(opt.param_groups) == 2
    by_wd = {g["weight_decay"]: g for g in opt.param_groups}
    assert set(by_wd) == {0.1, 0.0}

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    decayed = {id_to_name[id(p)] for p in by_wd[0.1]["params"]}
    not_decayed = {id_to_name[id(p)] for p in by_wd[0.0]["params"]}

    # Biases, embeddings, and normalization (RMSNorm) weights must NOT be weight-decayed.
    assert not any("bias" in n for n in decayed)
    assert not any("embed" in n for n in decayed)
    assert not any("norm" in n.lower() for n in decayed)
    # ...and they are present in the no-decay group.
    assert any("bias" in n for n in not_decayed)
    assert any("embed" in n for n in not_decayed)
    assert any("norm" in n.lower() for n in not_decayed)

    # The two groups exactly partition every trainable parameter.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert decayed.isdisjoint(not_decayed)
    assert decayed | not_decayed == trainable


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------
def _tokenized_parquet(path, token_ids):
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
            pa.field("age_normalized", pa.list_(pa.float32())),
        ]
    )
    n = len(token_ids)
    row = {
        "subject_id": 1,
        "index_time": datetime(2023, 1, 1),
        "token_ids": pa.array(token_ids, type=pa.int32()),
        "age": pa.array([float(i) for i in range(n)], type=pa.float32()),
        "length": n,
        "age_normalized": pa.array([0.0] * n, type=pa.float32()),
    }
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def test_prepare_data_returns_datasets_and_auto_computes_max_length(tmp_path):
    path = tmp_path / "tok.parquet"
    _tokenized_parquet(path, list(range(20)))  # one 20-token patient

    train_ds, train_sampler, val_ds, val_sampler = prepare_data(
        train_path=path,
        val_path=path,
        max_tokens_per_batch=16,
        min_patients_per_batch=2,
        max_seq_length_per_patient=None,  # -> auto-calc 16 // 2 = 8
        token_dropout_prob=0.0,
        min_patient_length=2,
        stride=8,
        num_ntp_classes=50,
    )

    assert isinstance(train_ds, TokenizedDataset)
    assert isinstance(val_ds, TokenizedDataset)
    assert isinstance(train_sampler, TokenBudgetBatchSampler)
    assert isinstance(val_sampler, TokenBudgetBatchSampler)
    # max_length auto-calc'd to 8: a length-20 seq with stride 8 yields windows at 12/4/0.
    assert len(train_ds) == 3


# ---------------------------------------------------------------------------
# prepare_model
# ---------------------------------------------------------------------------
def _tiny_lookup(dir_path, n=4, dim=8):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "id_mapping.json").write_text(json.dumps({f"t{i}": i for i in range(n)}))
    save_file(
        {"embedding_table": np.zeros((n, dim), dtype=np.float16)},
        str(dir_path / "embedding_table.safetensors"),
    )
    (dir_path / "metadata.json").write_text(json.dumps({"model_name": "test"}))
    return dir_path


def _args(**overrides):
    base = dict(
        vocab_size=50,
        hidden_size=16,
        n_layers=2,
        n_heads=2,
        attention_width=8,
        use_normed_ages=False,
        intermediate_size=24,
        use_bias=False,
        hidden_act="gelu",
        remove_first_block_norm=True,
        alternating_dense_layers=False,
        dense_every_n_layers=2,
        rope_base_sparse=100.0,
        rope_base_global=10000.0,
        separate_rope_by_attention=False,
        input_mode="discrete",
        num_ntp_classes=50,
        initial_weights_path=None,
        embedding_lookup_path=None,
        use_numerical_path=True,
        numerical_hidden_dim=8,
        numeric_pathway_mode="ref_range_priority",
        numerical_input_dim=None,
        freeze_text_embedding=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_prepare_model_discrete_builds_ehrfm():
    model = prepare_model(_args(input_mode="discrete"), _DEVICE, _LOGGER)
    assert isinstance(model, EHRFM)
    assert model.config.transformer.input_mode == "discrete"
    assert model.training  # prepare_model leaves the model in train mode


def test_prepare_model_embedding_refrange_derives_4dim(tmp_path):
    look = _tiny_lookup(tmp_path / "lookup")
    model = prepare_model(
        _args(
            input_mode="embedding", embedding_lookup_path=str(look), numeric_pathway_mode="ref_range_priority"
        ),
        _DEVICE,
        _LOGGER,
    )
    assert model.config.transformer.numerical_input_dim == 4
    assert isinstance(model.transformer.embed, DualPathInputEncoder)


def test_prepare_model_explicit_numerical_input_dim_overrides(tmp_path):
    look = _tiny_lookup(tmp_path / "lookup")
    model = prepare_model(
        _args(
            input_mode="embedding",
            embedding_lookup_path=str(look),
            numeric_pathway_mode="ref_range_priority",
            numerical_input_dim=7,  # conflicts with refrange's 4 -> explicit wins
        ),
        _DEVICE,
        _LOGGER,
    )
    assert model.config.transformer.numerical_input_dim == 7
