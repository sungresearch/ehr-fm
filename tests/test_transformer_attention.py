"""Behavioral tests for the DenseTransformer attention/positioning contracts.

Covers properties the rest of the suite relies on but never asserts:
1. Patients packed into one flat batch do NOT attend across each other
   (block-diagonal masking). A leak here would silently corrupt every
   representation and training step.
2. Local attention honors attention_width (a token is unaffected by tokens
   outside its window).
3. RoPE positions are derived from age-in-days, not token index.
"""

import torch

from ehr_fm.models.config import EHRFMConfig
from ehr_fm.models.transformer import EHRFM, _fixed_pos_embedding, packed_ehr_collate


def _discrete_model(n_layers=2, attention_width=8, alternating=True):
    cfg = EHRFMConfig(
        transformer={
            "vocab_size": 60,
            "hidden_size": 16,
            "intermediate_size": 24,
            "n_heads": 2,
            "n_layers": n_layers,
            "attention_width": attention_width,
            "input_mode": "discrete",
            "alternating_dense_layers": alternating,
            "dense_every_n_layers": 2,
        },
        task={"task_type": "sequence_classification", "n_classes": 60},
    )
    return EHRFM(cfg).eval()


def _packed(patients):
    """Build a packed batch from a list of token-id lists (ages restart per patient)."""
    batch = []
    for i, toks in enumerate(patients):
        n = len(toks)
        batch.append(
            {
                "input_ids": torch.tensor(toks, dtype=torch.long),
                "labels": torch.full((n,), -100, dtype=torch.long),
                "age": torch.arange(n, dtype=torch.float32),
                "age_normalized": torch.zeros(n, dtype=torch.float32),
                "length": n,
                "patient_id": i,
                "index_time": float(i),
            }
        )
    return packed_ehr_collate(batch)


class TestCrossPatientIsolation:
    def test_packed_patients_match_running_each_alone(self):
        """No cross-patient attention: each patient's packed hidden states equal
        the states it produces when run by itself."""
        torch.manual_seed(0)
        model = _discrete_model()
        p1 = [1, 2, 3, 4, 5]
        p2 = [10, 11, 12]
        with torch.no_grad():
            packed = model.transformer(_packed([p1, p2]))
            alone1 = model.transformer(_packed([p1]))
            alone2 = model.transformer(_packed([p2]))
        assert torch.allclose(packed[: len(p1)], alone1, atol=1e-5)
        assert torch.allclose(packed[len(p1) :], alone2, atol=1e-5)


class TestLocalAttentionWidth:
    def test_token_outside_window_has_no_effect(self):
        """With a single all-local layer, a token outside the local window cannot
        influence a later token's output."""
        torch.manual_seed(0)
        model = _discrete_model(n_layers=1, attention_width=2, alternating=False)
        toks = [1, 2, 3, 4, 5, 6, 7, 8]
        with torch.no_grad():
            base = model.transformer(_packed([toks]))
            perturbed = model.transformer(_packed([[50, 2, 3, 4, 5, 6, 7, 8]]))
        # Last token is far outside the first token's window -> unaffected.
        assert torch.allclose(base[-1], perturbed[-1], atol=1e-6)
        # The perturbed token itself changes (sanity that the perturbation took effect).
        assert not torch.allclose(base[0], perturbed[0], atol=1e-6)


class TestRoPEAgeCalibration:
    def test_positions_derive_from_age_not_index(self):
        dim = 8
        ages = torch.tensor([5.0, 5.0, 10.0, 0.0])
        sin, cos = _fixed_pos_embedding(ages, dim, torch.float32, base=100.0)

        assert sin.shape == (4, 1, dim)
        assert cos.shape == (4, 1, dim)
        # Same age at different sequence positions -> identical encoding (age, not index).
        assert torch.allclose(sin[0], sin[1])
        assert torch.allclose(cos[0], cos[1])
        # Age 0 -> sin=0, cos=1.
        assert torch.allclose(sin[3], torch.zeros_like(sin[3]))
        assert torch.allclose(cos[3], torch.ones_like(cos[3]))
        # Different age -> different encoding.
        assert not torch.allclose(sin[0], sin[2])
