"""Tests for train_ehr_fm helpers: build_optimizer, prepare_data, prepare_model."""

import copy
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file as load_safetensors
from transformers import get_scheduler

from ehr_fm.data import TokenBudgetBatchSampler, TokenizedDataset
from ehr_fm.embedding.lookup import TiedEmbedding
from ehr_fm.logger import setup_logging
from ehr_fm.models import DualPathInputEncoder
from ehr_fm.models.config import EHRFMConfig
from ehr_fm.models.transformer import EHRFM
from ehr_fm.scripts.compute_representations import _load_encoder_weights
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
def _tiny_lookup(dir_path, n=4, dim=8, table=None, row_index=None):
    """A lookup of zeros, or of ``table``; with ``row_index``, string i reads row ``row_index[i]`` of it."""
    table = np.zeros((n, dim)) if table is None else table
    n_ids = len(table) if row_index is None else len(row_index)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "id_mapping.json").write_text(json.dumps({f"t{i}": i for i in range(n_ids)}))
    save_file(
        {"embedding_table": table.astype(np.float16)},
        str(dir_path / "embedding_table.safetensors"),
    )
    if row_index is not None:
        save_file({"row_index": row_index.astype(np.int64)}, str(dir_path / "row_index.safetensors"))
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
        seed=42,
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


def _initial_state(tmp_path, **overrides):
    look = _tiny_lookup(tmp_path / "lookup")
    args = _args(input_mode="embedding", embedding_lookup_path=str(look), **overrides)
    return prepare_model(args, _DEVICE, _LOGGER).state_dict()


def test_prepare_model_initial_weights_follow_the_seed(tmp_path):
    first, again, other = (_initial_state(tmp_path, seed=seed) for seed in (42, 42, 43))
    assert all(torch.equal(first[k], again[k]) for k in first)
    key = "transformer.layers.0.input_proj.weight"
    assert not torch.equal(first[key], other[key])


def test_prepare_model_only_the_task_head_changes_with_its_size(tmp_path):
    small, large = (_initial_state(tmp_path, num_ntp_classes=n) for n in (50, 60))
    shared = [k for k in small if not k.startswith("task_head.")]
    assert any(k.startswith("transformer.embed.") for k in shared)  # the encoder, built before the model
    assert all(torch.equal(small[k], large[k]) for k in shared)


# ---------------------------------------------------------------------------
# Trainable text embedding (--no-freeze_text_embedding)
# ---------------------------------------------------------------------------
_OPTIMIZER = {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}


def _batch(text_ids):
    n = len(text_ids)
    return {
        "input_ids": torch.arange(1, n + 1),
        "embedding_text_ids": torch.tensor(text_ids),
        "numeric_features": torch.zeros(n, 4),
        "ages": torch.arange(n, dtype=torch.float32),
        "normalized_ages": torch.arange(n, dtype=torch.float32) / 10,
        "patient_lengths": torch.tensor([n], dtype=torch.int32),
        "label_indices": torch.arange(n - 1),
        "task": {"labels": torch.arange(2, n + 1)},
    }


def _table_model(look, freeze=False):
    args = _args(input_mode="embedding", embedding_lookup_path=str(look), freeze_text_embedding=freeze)
    return prepare_model(args, _DEVICE, _LOGGER)


def test_unfrozen_text_embedding_trains_without_weight_decay(tmp_path):
    model = _table_model(_tiny_lookup(tmp_path / "lookup", n=6))
    table = model.transformer.embed.text_embedding.weight
    optimizer = build_optimizer(config=_OPTIMIZER, model=model, logger=_LOGGER)
    no_decay = optimizer.param_groups[1]
    assert no_decay["weight_decay"] == 0.0 and any(p is table for p in no_decay["params"])

    before = table.detach().clone()
    model(_batch([0, 1, 3, 1, 0]))[0].backward()
    optimizer.step()
    after = table.detach()
    assert after[[0, 1, 3]].ne(before[[0, 1, 3]]).any(dim=1).all()  # the rows read
    assert torch.equal(after[[2, 4, 5]], before[[2, 4, 5]])


# ---------------------------------------------------------------------------
# Text embedding with shared rows (a lookup with row_index.safetensors)
# ---------------------------------------------------------------------------
_ROWS = np.random.default_rng(0).standard_normal((3, 8))
_ROW_INDEX = np.array([0, 2, 0, 1, 2, 1])  # six strings on three rows
_TABLE_PREFIX = "transformer.embed.text_embedding."


def test_tied_text_embedding_trains_one_row_per_group_and_is_checkpointed(tmp_path):
    model = _table_model(_tiny_lookup(tmp_path / "tied", table=_ROWS, row_index=_ROW_INDEX))
    table = model.transformer.embed.text_embedding
    assert isinstance(table, TiedEmbedding)

    optimizer = build_optimizer(config=_OPTIMIZER, model=model, logger=_LOGGER)
    model(_batch([0, 1, 3, 1, 0]))[0].backward()  # reads rows 0, 2 and 1 through strings 0, 1 and 3
    optimizer.step()
    rows = table(torch.arange(6)).detach()  # strings 0 and 2 share row 0, 1 and 4 row 2, 3 and 5 row 1
    assert torch.equal(rows[0], rows[2]) and torch.equal(rows[1], rows[4]) and torch.equal(rows[3], rows[5])
    assert not torch.equal(rows[0], torch.from_numpy(_ROWS[0].astype(np.float16).astype(np.float32)))

    model.save_pretrained(tmp_path / "checkpoint")
    state = load_safetensors(str(tmp_path / "checkpoint" / "model.safetensors"))
    assert torch.equal(state[f"{_TABLE_PREFIX}weight"], table.weight.detach())
    assert torch.equal(state[f"{_TABLE_PREFIX}row_index"], torch.from_numpy(_ROW_INDEX))

    # Extraction reads the trained table from an exported plain lookup and skips the checkpoint's table keys.
    extraction = _table_model(_tiny_lookup(tmp_path / "export", table=rows.numpy()), freeze=True)
    _load_encoder_weights(extraction, tmp_path / "checkpoint", _LOGGER)
    for name, value in model.transformer.embed.state_dict().items():
        if not name.startswith("text_embedding."):
            assert torch.equal(extraction.transformer.embed.state_dict()[name], value)


def test_tied_and_unfrozen_text_embeddings_keep_every_other_initial_value(tmp_path):
    def outside_and_table(model):
        state = {k: v for k, v in model.state_dict().items() if not k.startswith(_TABLE_PREFIX)}
        return state, model.transformer.embed.text_embedding(torch.arange(6)).detach()

    plain = _tiny_lookup(tmp_path / "plain", table=_ROWS[_ROW_INDEX])
    tied = _tiny_lookup(tmp_path / "tied", table=_ROWS, row_index=_ROW_INDEX)
    frozen, frozen_table = outside_and_table(_table_model(plain, freeze=True))
    for model in (_table_model(plain), _table_model(tied)):
        state, table = outside_and_table(model)
        assert state.keys() == frozen.keys() and all(torch.equal(state[k], frozen[k]) for k in frozen)
        assert torch.equal(table, frozen_table)


# ---------------------------------------------------------------------------
# Learning rate of the text embedding table (--text_embedding_lr_scale)
# ---------------------------------------------------------------------------
_TABLE = f"{_TABLE_PREFIX}weight"
_SCALE = 0.1


def _scaled(scale):
    return {**_OPTIMIZER, "text_embedding_lr_scale": scale}


def _groups(config, model):
    """(names in order, lr, weight_decay) of each parameter group of the optimizer that ``config`` gives."""
    optimizer = build_optimizer(config=config, model=model, logger=_LOGGER)
    names = {id(p): n for n, p in model.named_parameters()}
    return [([names[id(p)] for p in g["params"]], g["lr"], g["weight_decay"]) for g in optimizer.param_groups]


def test_text_embedding_lr_scale_gives_the_table_its_own_group(tmp_path):
    model = _table_model(_tiny_lookup(tmp_path / "lookup", table=_ROWS[_ROW_INDEX]))
    decay, no_decay = _groups(_OPTIMIZER, model)
    assert _TABLE in no_decay[0]
    assert _groups(_scaled(1.0), model) == [decay, no_decay]

    others = ([n for n in no_decay[0] if n != _TABLE], *no_decay[1:])
    assert _groups(_scaled(_SCALE), model) == [decay, others, ([_TABLE], _SCALE * _OPTIMIZER["lr"], 0.0)]


def test_text_embedding_lr_scale_follows_the_schedule(tmp_path):
    model = _table_model(_tiny_lookup(tmp_path / "lookup", table=_ROWS[_ROW_INDEX]))
    optimizer = build_optimizer(config=_scaled(_SCALE), model=model, logger=_LOGGER)
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer,
        num_warmup_steps=150,
        num_training_steps=1200,
        scheduler_specific_kwargs={"min_lr_rate": 0.1},
    )
    rates = {}
    for step in range(1200):
        if step in (0, 150, 1000, 1199):
            rates[step] = [g["lr"] for g in optimizer.param_groups]
        optimizer.step()  # no gradients, so no parameter changes
        scheduler.step()
    assert rates[150][0] == pytest.approx(_OPTIMIZER["lr"])
    assert rates[1199][0] < rates[1000][0] < rates[150][0]
    for decay, no_decay, table in rates.values():
        assert decay == no_decay and table == pytest.approx(_SCALE * decay, rel=1e-12, abs=0.0)


def test_text_embedding_lr_scale_scales_the_table_step_only(tmp_path):
    model = _table_model(_tiny_lookup(tmp_path / "lookup", table=_ROWS[_ROW_INDEX]))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    after = {}
    for scale in (1.0, _SCALE):
        run = copy.deepcopy(model)
        optimizer = build_optimizer(config=_scaled(scale), model=run, logger=_LOGGER)
        torch.manual_seed(0)
        run(_batch([0, 1, 3, 1, 0]))[0].backward()
        optimizer.step()
        after[scale] = {n: p.detach().clone() for n, p in run.named_parameters()}

    for name in before:
        if name != _TABLE:
            assert torch.equal(after[_SCALE][name], after[1.0][name]), name
    start = before[_TABLE].double()
    step_one, step_scaled = after[1.0][_TABLE].double() - start, after[_SCALE][_TABLE].double() - start
    assert step_one[[0, 1, 3]].ne(0).any(dim=1).all() and step_one[[2, 4, 5]].eq(0).all()
    # The table is stored in float32: each step includes one rounding of the parameter, at most eps * |p| / 2.
    bound = torch.finfo(torch.float32).eps * start.abs()
    assert ((step_scaled - _SCALE * step_one).abs() <= bound).all()


def test_text_embedding_lr_scale_needs_a_trainable_table_and_a_positive_scale(tmp_path):
    look = _tiny_lookup(tmp_path / "lookup", table=_ROWS[_ROW_INDEX])
    for model in (_table_model(look, freeze=True), _model()):
        with pytest.raises(ValueError, match="needs a trainable text embedding table"):
            build_optimizer(config=_scaled(_SCALE), model=model, logger=_LOGGER)
    for scale in (0.0, -0.1):
        with pytest.raises(ValueError, match="must be positive"):
            build_optimizer(config=_scaled(scale), model=_table_model(look), logger=_LOGGER)
