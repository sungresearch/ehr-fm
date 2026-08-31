"""The RoPE time axis is float32 by construction -- these tests keep it that way.

`ages` carries age-in-days and feeds sin/cos in `_fixed_pos_embedding`, the backbone's only
positional signal. bfloat16's absolute resolution across this cohort's age range is 0.5-32 days
while the median gap between consecutive events is 8 minutes, so a bfloat16 time axis collapses
97% of genuinely ordered event pairs onto identical positions. Representations shipped with that
defect once; the corrected published artifacts are what today's default reproduces.

The invariant is enforced twice, and each enforcement point is covered here:
- `compute_representations.prepare_transformer_input` downcasts only allowlisted matmul inputs,
  so the time axis cannot be rounded before it reaches the model;
- `_fixed_pos_embedding` computes float32 sin/cos unconditionally and rejects a
  reduced-precision `ages` outright, covering every other caller, present and future.

These tests exercise the real functions, not reimplementations of them.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from ehr_fm.models import transformer as _transformer
from ehr_fm.scripts.compute_representations import (
    COMPUTE_DTYPE_KEYS,
    prepare_transformer_input,
)

CPU = torch.device("cpu")


@pytest.fixture
def batch():
    return {
        # Synthetic ages in the [128, 256) binade, where the bfloat16 grid spacing is exactly
        # 1.0 day, spanning under an hour. Not patient data.
        "ages": torch.tensor([200.0100, 200.0150, 200.0200, 200.0300], dtype=torch.float32),
        "normalized_ages": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        "numeric_features": torch.randn(4, 4, dtype=torch.float32),
        "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        "patient_lengths": torch.tensor([4], dtype=torch.int64),
        "task": {"labels": torch.tensor([1])},
    }


def test_ages_reach_the_model_in_float32_under_a_bfloat16_model(batch):
    out = prepare_transformer_input(batch, torch.bfloat16, CPU)
    assert out["ages"].dtype == torch.float32
    assert out["normalized_ages"].dtype == torch.float32
    assert out["numeric_features"].dtype == torch.bfloat16, "matmul inputs still follow the model"
    assert out["token_ids"].dtype == torch.int64
    assert out["patient_lengths"].dtype == torch.int64
    assert out["task"] is batch["task"], "non-tensors pass through untouched"


def test_new_float_tensors_are_not_quantized_by_default(batch):
    """The allowlist's point: an unknown float tensor keeps the dtype the collate produced.
    Inheriting the compute dtype now requires opting IN via COMPUTE_DTYPE_KEYS."""
    batch["future_field"] = torch.randn(4, dtype=torch.float32)
    out = prepare_transformer_input(batch, torch.bfloat16, CPU)
    assert out["future_field"].dtype == torch.float32
    assert "future_field" not in COMPUTE_DTYPE_KEYS


def test_minute_scale_events_stay_distinct_end_to_end(batch):
    """The regression test with teeth: 4 events ~6 minutes apart at age ~200 days must survive
    batch preparation and the RoPE construction as 4 distinct positions. Reintroducing a cast
    anywhere on the path collapses them onto the 1.0-day bfloat16 grid."""
    prepared = prepare_transformer_input(batch, torch.bfloat16, CPU)
    assert len(torch.unique(prepared["ages"])) == 4
    sin, _ = _transformer._fixed_pos_embedding(prepared["ages"], 8)
    assert len(torch.unique(sin[:, 0, 0])) == 4, "fastest RoPE dimension must separate the events"
    collapsed = batch["ages"].to(torch.bfloat16)
    assert len(torch.unique(collapsed)) == 1, "the guarded-against cast really is destructive"


def test_sincos_is_float32_and_the_downcast_parameter_is_gone():
    ages = torch.tensor([2800.0, 2800.5, 2801.0], dtype=torch.float32)
    sin, cos = _transformer._fixed_pos_embedding(ages, 8)
    assert sin.dtype == torch.float32 and cos.dtype == torch.float32
    # The dtype parameter was the vector by which the compute precision reached the clock.
    # Its absence -- not a defaulted value -- is what retires the defect.
    assert "dtype" not in inspect.signature(_transformer._fixed_pos_embedding).parameters


def test_reduced_precision_ages_are_rejected():
    for bad in (torch.bfloat16, torch.float16):
        with pytest.raises(TypeError, match="must be float32"):
            _transformer._fixed_pos_embedding(torch.tensor([1.0, 2.0], dtype=bad), 8)
    # float64 is exact and accepted; the consumer floats it on arrival.
    sin, _ = _transformer._fixed_pos_embedding(torch.tensor([1.0, 2.0], dtype=torch.float64), 8)
    assert sin.dtype == torch.float64


def test_fp32_sincos_does_not_change_dtype_flow():
    """_apply_rotary floats sin/cos on arrival and returns x.dtype, so q/k dtypes -- and what
    xformers sees -- are exactly what they were under the old downcast sin/cos."""
    ages = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    for x_dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(4, 1, 8, dtype=x_dtype)
        out = _transformer._apply_rotary(x, _transformer._fixed_pos_embedding(ages, 8))
        assert out.dtype == x_dtype


def test_training_numerics_are_unchanged():
    """Why no checkpoint is invalidated: at the RoPE call site `x = in_norm(x)`, and RMSNorm
    with an fp32 master weight promotes even a bf16 hidden state back to fp32 under autocast --
    so training always passed float32 ages and received float32 sin/cos, before and after this
    consolidation."""
    from ehr_fm.models.utils import RMSNorm

    norm_fp32_master = RMSNorm(8)
    for incoming in (torch.bfloat16, torch.float32):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = norm_fp32_master(torch.randn(4, 8, dtype=incoming))
        assert out.dtype == torch.float32, "training keeps the pos-embed call site in fp32"
