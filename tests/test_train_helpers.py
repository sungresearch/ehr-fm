"""Tests for train_ehr_fm helpers -- build_optimizer weight-decay grouping.

A bug in the param grouping silently mis-regularizes the model (e.g. decaying the
embedding table), so we pin the invariants that matter: biases and embeddings are
excluded from weight decay, and the two groups exactly partition the trainable params.
"""

from ehr_fm.logger import setup_logging
from ehr_fm.models.config import EHRFMConfig
from ehr_fm.models.transformer import EHRFM
from ehr_fm.scripts.train_ehr_fm import build_optimizer


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
        logger=setup_logging(),
    )

    assert len(opt.param_groups) == 2
    by_wd = {g["weight_decay"]: g for g in opt.param_groups}
    assert set(by_wd) == {0.1, 0.0}

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    decayed = {id_to_name[id(p)] for p in by_wd[0.1]["params"]}
    not_decayed = {id_to_name[id(p)] for p in by_wd[0.0]["params"]}

    # Biases and embeddings must NOT be weight-decayed.
    assert not any("bias" in n for n in decayed)
    assert not any("embed" in n for n in decayed)
    # ...and they are present in the no-decay group.
    assert any("bias" in n for n in not_decayed)
    assert any("embed" in n for n in not_decayed)

    # The two groups exactly partition every trainable parameter.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert decayed.isdisjoint(not_decayed)
    assert decayed | not_decayed == trainable
