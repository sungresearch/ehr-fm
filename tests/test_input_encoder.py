"""Tests for the dual-path input encoder (Arm A1)."""

import torch
from torch import nn

from ehr_fm.models.input_encoder import (
    DualPathInputEncoder,
    NumericalEncoder,
    TextProjection,
)


class TestTextProjection:
    def test_output_shape(self):
        proj = TextProjection(embedding_dim=4096, hidden_size=768)
        x = torch.randn(10, 4096)
        out = proj(x)
        assert out.shape == (10, 768)

    def test_gradient_flow(self):
        proj = TextProjection(embedding_dim=128, hidden_size=64)
        x = torch.randn(5, 128, requires_grad=True)
        out = proj(x)
        out.sum().backward()
        assert x.grad is not None


class TestNumericalEncoder:
    def test_output_shapes(self):
        enc = NumericalEncoder(input_dim=5, hidden_dim=128, output_dim=768)
        x = torch.randn(10, 5)
        gamma, beta = enc(x)
        assert gamma.shape == (10, 768)
        assert beta.shape == (10, 768)

    def test_output_shapes_dim4(self):
        enc = NumericalEncoder(input_dim=4, hidden_dim=128, output_dim=768)
        x = torch.randn(10, 4)
        gamma, beta = enc(x)
        assert gamma.shape == (10, 768)
        assert beta.shape == (10, 768)

    def test_identity_init(self):
        """At init, gamma should be ~1 and beta should be ~0."""
        enc = NumericalEncoder(input_dim=5, hidden_dim=128, output_dim=768)
        x = torch.zeros(1, 5)
        gamma, beta = enc(x)
        # With zero input and proper init, gamma_head(MLP(0)) should be close to 1
        # beta_head(MLP(0)) should be close to 0
        # The MLP output at zero input will be non-zero due to GELU,
        # but the gamma/beta heads have zero weights, so output comes from bias only
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.1)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=0.1)

    def test_identity_init_dim4(self):
        """Same identity init property with ref_range_priority feature dim."""
        enc = NumericalEncoder(input_dim=4, hidden_dim=128, output_dim=768)
        x = torch.zeros(1, 4)
        gamma, beta = enc(x)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.1)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=0.1)

    def test_hard_gate_dim4(self):
        """value_present=0 (last element) should produce identity modulation."""
        enc = NumericalEncoder(input_dim=4, hidden_dim=64, output_dim=32)
        # [x_primary=0.5, is_refrange=1, is_log1p=0, value_present=0]
        x = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        gamma, beta = enc(x)
        assert torch.allclose(gamma, torch.ones_like(gamma))
        assert torch.allclose(beta, torch.zeros_like(beta))

    def test_output_shapes_dim15(self):
        """fourier_ref_range_priority: 15-dim input."""
        enc = NumericalEncoder(input_dim=15, hidden_dim=128, output_dim=768)
        x = torch.randn(10, 15)
        gamma, beta = enc(x)
        assert gamma.shape == (10, 768)
        assert beta.shape == (10, 768)

    def test_identity_init_dim15(self):
        """Same identity init property with fourier_ref_range_priority feature dim."""
        enc = NumericalEncoder(input_dim=15, hidden_dim=128, output_dim=768)
        x = torch.zeros(1, 15)
        gamma, beta = enc(x)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.1)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=0.1)

    def test_hard_gate_dim15_absent(self):
        """Absent event: value_present=0 (last element) -> identity."""
        enc = NumericalEncoder(input_dim=15, hidden_dim=64, output_dim=32)
        x = torch.zeros(1, 15)  # all zeros including value_present=0
        gamma, beta = enc(x)
        assert torch.allclose(gamma, torch.ones_like(gamma))
        assert torch.allclose(beta, torch.zeros_like(beta))

    def test_hard_gate_dim15_present(self):
        """Present event: value_present=1 (last element) -> MLP modulates."""
        enc = NumericalEncoder(input_dim=15, hidden_dim=64, output_dim=32)
        # Fourier features + [is_refrange=1, is_log1p=0, value_present=1]
        x = torch.randn(1, 15)
        x[0, -3] = 1.0  # is_refrange
        x[0, -2] = 0.0  # is_log1p
        x[0, -1] = 1.0  # value_present
        gamma, beta = enc(x)
        # At init, gamma ~1 and beta ~0 but not exactly gated to identity
        assert gamma.shape == (1, 32)


class TestDualPathInputEncoder:
    def _make_encoder(
        self, num_embeddings=100, embedding_dim=128, hidden_size=64,
        use_numerical=True, numerical_input_dim=5,
    ):
        text_emb = nn.Embedding(num_embeddings, embedding_dim)
        text_emb.weight.requires_grad = False  # frozen
        return DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=hidden_size,
            use_numerical_path=use_numerical,
            numerical_input_dim=numerical_input_dim,
            numerical_hidden_dim=32,
        )

    def test_dual_path_output_shape(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 100, (10,))
        feats = torch.randn(10, 5)
        out = enc(ids, feats)
        assert out.shape == (10, 64)

    def test_dual_path_output_shape_dim4(self):
        enc = self._make_encoder(numerical_input_dim=4)
        ids = torch.randint(0, 100, (10,))
        feats = torch.randn(10, 4)
        out = enc(ids, feats)
        assert out.shape == (10, 64)

    def test_dual_path_output_shape_dim15(self):
        enc = self._make_encoder(numerical_input_dim=15)
        ids = torch.randint(0, 100, (10,))
        feats = torch.randn(10, 15)
        out = enc(ids, feats)
        assert out.shape == (10, 64)

    def test_text_only_mode(self):
        enc = self._make_encoder(use_numerical=False)
        ids = torch.randint(0, 100, (10,))
        out = enc(ids, None)
        assert out.shape == (10, 64)
        # Should not have numerical_encoder attribute
        assert not hasattr(enc, "numerical_encoder")

    def test_text_only_no_numeric_features(self):
        enc = self._make_encoder(use_numerical=True)
        ids = torch.randint(0, 100, (10,))
        # With use_numerical_path=True but no numeric_features, should still work
        out = enc(ids, None)
        assert out.shape == (10, 64)

    def test_frozen_text_embedding_no_grad(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 100, (5,))
        feats = torch.randn(5, 5)
        out = enc(ids, feats)
        out.sum().backward()

        # Frozen text embedding should have no grad
        assert enc.text_embedding.weight.grad is None

        # Projection should have grad
        for p in enc.text_projection.parameters():
            assert p.grad is not None

    def test_numerical_encoder_has_grad(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 100, (5,))
        feats = torch.randn(5, 5)
        out = enc(ids, feats)
        out.sum().backward()

        for p in enc.numerical_encoder.parameters():
            assert p.grad is not None


class TestEmbeddingCollation:
    """Test packed_ehr_collate with embedding mode fields."""

    def test_collate_with_embedding_fields(self):
        from ehr_fm.models.transformer import packed_ehr_collate

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([2, 3, -100]),
                "age": torch.tensor([0.0, 1.0, 2.0]),
                "age_normalized": torch.tensor([0.0, 0.1, 0.2]),
                "length": 3,
                "patient_id": 1,
                "index_time": 1000.0,
                "embedding_text_ids": torch.tensor([10, 20, 30]),
                "numeric_features": torch.randn(3, 5),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([5, -100]),
                "age": torch.tensor([0.0, 3.0]),
                "age_normalized": torch.tensor([0.0, 0.3]),
                "length": 2,
                "patient_id": 2,
                "index_time": 2000.0,
                "embedding_text_ids": torch.tensor([40, 50]),
                "numeric_features": torch.randn(2, 5),
            },
        ]

        collated = packed_ehr_collate(batch)

        assert "embedding_text_ids" in collated
        assert collated["embedding_text_ids"].shape == (5,)
        assert torch.equal(collated["embedding_text_ids"], torch.tensor([10, 20, 30, 40, 50]))

        assert "numeric_features" in collated
        assert collated["numeric_features"].shape == (5, 5)

    def test_collate_with_dim4_numeric_features(self):
        """ref_range_priority: 4-dim numeric features collate correctly."""
        from ehr_fm.models.transformer import packed_ehr_collate

        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([2, -100]),
                "age": torch.tensor([0.0, 1.0]),
                "age_normalized": torch.tensor([0.0, 0.1]),
                "length": 2,
                "patient_id": 1,
                "index_time": 1000.0,
                "embedding_text_ids": torch.tensor([10, 20]),
                "numeric_features": torch.randn(2, 4),
            },
        ]
        collated = packed_ehr_collate(batch)
        assert collated["numeric_features"].shape == (2, 4)

    def test_collate_with_dim15_numeric_features(self):
        """fourier_ref_range_priority: 15-dim numeric features collate correctly."""
        from ehr_fm.models.transformer import packed_ehr_collate

        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([2, -100]),
                "age": torch.tensor([0.0, 1.0]),
                "age_normalized": torch.tensor([0.0, 0.1]),
                "length": 2,
                "patient_id": 1,
                "index_time": 1000.0,
                "embedding_text_ids": torch.tensor([10, 20]),
                "numeric_features": torch.randn(2, 15),
            },
        ]
        collated = packed_ehr_collate(batch)
        assert collated["numeric_features"].shape == (2, 15)

    def test_collate_without_embedding_fields(self):
        """Backward compat: batch without embedding fields still works."""
        from ehr_fm.models.transformer import packed_ehr_collate

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([2, 3, -100]),
                "age": torch.tensor([0.0, 1.0, 2.0]),
                "age_normalized": torch.tensor([0.0, 0.1, 0.2]),
                "length": 3,
                "patient_id": 1,
                "index_time": 1000.0,
            },
        ]

        collated = packed_ehr_collate(batch)
        assert "embedding_text_ids" not in collated
        assert "numeric_features" not in collated
        assert collated["input_ids"].shape == (3,)


class TestEHRFMEmbeddingMode:
    """Integration tests for EHRFM in embedding mode."""

    def test_forward_embedding_mode(self):
        from ehr_fm.models.config import EHRFMConfig
        from ehr_fm.models.transformer import EHRFM

        num_embeddings = 50
        embedding_dim = 32
        hidden_size = 64

        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 100,
                "hidden_size": hidden_size,
                "intermediate_size": 96,
                "n_heads": 4,
                "n_layers": 2,
                "attention_width": 128,
                "input_mode": "embedding",
                "embedding_dim": embedding_dim,
                "use_numerical_path": True,
                "numerical_input_dim": 5,
                "numerical_hidden_dim": 16,
            },
            task={"task_type": "sequence_classification", "n_classes": 100},
        )

        model = EHRFM(cfg)

        # Wire up the input encoder
        text_emb = nn.Embedding(num_embeddings, embedding_dim)
        text_emb.weight.requires_grad = False
        encoder = DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=hidden_size,
            use_numerical_path=True,
            numerical_input_dim=5,
            numerical_hidden_dim=16,
        )
        model.transformer.set_input_encoder(encoder)

        batch = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "embedding_text_ids": torch.randint(0, num_embeddings, (5,)),
            "numeric_features": torch.randn(5, 5),
            "ages": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            "normalized_ages": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            "patient_lengths": torch.tensor([5], dtype=torch.int32),
            "label_indices": torch.tensor([0, 1, 2, 3]),
            "task": {"labels": torch.tensor([2, 3, 4, 5])},
        }

        loss, result = model(batch)
        assert loss.isfinite()
        assert loss.shape == ()

    def test_forward_ref_range_priority_mode(self):
        """Full forward pass with 4-dim numeric features (ref_range_priority)."""
        from ehr_fm.models.config import EHRFMConfig
        from ehr_fm.models.transformer import EHRFM

        num_embeddings = 50
        embedding_dim = 32
        hidden_size = 64

        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 100,
                "hidden_size": hidden_size,
                "intermediate_size": 96,
                "n_heads": 4,
                "n_layers": 2,
                "attention_width": 128,
                "input_mode": "embedding",
                "embedding_dim": embedding_dim,
                "use_numerical_path": True,
                "numerical_input_dim": 4,
                "numerical_hidden_dim": 16,
                "numeric_pathway_mode": "ref_range_priority",
            },
            task={"task_type": "sequence_classification", "n_classes": 100},
        )

        model = EHRFM(cfg)

        text_emb = nn.Embedding(num_embeddings, embedding_dim)
        text_emb.weight.requires_grad = False
        encoder = DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=hidden_size,
            use_numerical_path=True,
            numerical_input_dim=4,
            numerical_hidden_dim=16,
        )
        model.transformer.set_input_encoder(encoder)

        batch = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "embedding_text_ids": torch.randint(0, num_embeddings, (5,)),
            "numeric_features": torch.randn(5, 4),
            "ages": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            "normalized_ages": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            "patient_lengths": torch.tensor([5], dtype=torch.int32),
            "label_indices": torch.tensor([0, 1, 2, 3]),
            "task": {"labels": torch.tensor([2, 3, 4, 5])},
        }

        loss, result = model(batch)
        assert loss.isfinite()
        assert loss.shape == ()

    def test_forward_fourier_ref_range_priority_mode(self):
        """Full forward pass with 15-dim numeric features (fourier_ref_range_priority)."""
        from ehr_fm.models.config import EHRFMConfig
        from ehr_fm.models.transformer import EHRFM

        num_embeddings = 50
        embedding_dim = 32
        hidden_size = 64

        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 100,
                "hidden_size": hidden_size,
                "intermediate_size": 96,
                "n_heads": 4,
                "n_layers": 2,
                "attention_width": 128,
                "input_mode": "embedding",
                "embedding_dim": embedding_dim,
                "use_numerical_path": True,
                "numerical_input_dim": 15,
                "numerical_hidden_dim": 16,
                "numeric_pathway_mode": "fourier_ref_range_priority",
            },
            task={"task_type": "sequence_classification", "n_classes": 100},
        )

        model = EHRFM(cfg)

        text_emb = nn.Embedding(num_embeddings, embedding_dim)
        text_emb.weight.requires_grad = False
        encoder = DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=hidden_size,
            use_numerical_path=True,
            numerical_input_dim=15,
            numerical_hidden_dim=16,
        )
        model.transformer.set_input_encoder(encoder)

        batch = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "embedding_text_ids": torch.randint(0, num_embeddings, (5,)),
            "numeric_features": torch.randn(5, 15),
            "ages": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            "normalized_ages": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            "patient_lengths": torch.tensor([5], dtype=torch.int32),
            "label_indices": torch.tensor([0, 1, 2, 3]),
            "task": {"labels": torch.tensor([2, 3, 4, 5])},
        }

        loss, result = model(batch)
        assert loss.isfinite()
        assert loss.shape == ()

    def test_forward_text_only_mode(self):
        from ehr_fm.models.config import EHRFMConfig
        from ehr_fm.models.transformer import EHRFM

        num_embeddings = 50
        embedding_dim = 32
        hidden_size = 64

        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 100,
                "hidden_size": hidden_size,
                "intermediate_size": 96,
                "n_heads": 4,
                "n_layers": 2,
                "attention_width": 128,
                "input_mode": "embedding",
                "embedding_dim": embedding_dim,
                "use_numerical_path": False,
            },
            task={"task_type": "sequence_classification", "n_classes": 100},
        )

        model = EHRFM(cfg)

        text_emb = nn.Embedding(num_embeddings, embedding_dim)
        text_emb.weight.requires_grad = False
        encoder = DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=hidden_size,
            use_numerical_path=False,
        )
        model.transformer.set_input_encoder(encoder)

        batch = {
            "input_ids": torch.tensor([1, 2, 3]),
            "embedding_text_ids": torch.randint(0, num_embeddings, (3,)),
            "ages": torch.tensor([0.0, 1.0, 2.0]),
            "normalized_ages": torch.tensor([0.0, 0.1, 0.2]),
            "patient_lengths": torch.tensor([3], dtype=torch.int32),
            "label_indices": torch.tensor([0, 1]),
            "task": {"labels": torch.tensor([2, 3])},
        }

        loss, result = model(batch)
        assert loss.isfinite()

    def test_discrete_mode_regression(self):
        """Verify discrete mode still works identically."""
        from ehr_fm.models.config import EHRFMConfig
        from ehr_fm.models.transformer import EHRFM

        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 100,
                "hidden_size": 64,
                "intermediate_size": 96,
                "n_heads": 4,
                "n_layers": 2,
                "attention_width": 128,
                "input_mode": "discrete",
            },
            task={"task_type": "sequence_classification", "n_classes": 100},
        )

        model = EHRFM(cfg)

        batch = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "ages": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            "normalized_ages": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            "patient_lengths": torch.tensor([5], dtype=torch.int32),
            "label_indices": torch.tensor([0, 1, 2, 3]),
            "task": {"labels": torch.tensor([2, 3, 4, 5])},
        }

        loss, result = model(batch)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Tests for the n_layers / activation / combiner knobs.
#
# Defaults must reproduce pre-knob behavior byte-identically (state_dict
# layout, forward computation). Non-default settings open up new structures
# and combiner classes.
# ---------------------------------------------------------------------------

import pytest

from ehr_fm.models.input_encoder import (
    AddCombiner,
    ConcatCombiner,
    _build_numeric_mlp,
)


class TestNumericMLPBuilder:
    def test_default_matches_legacy_structure(self):
        """n_layers=2, activation='gelu' produces 2 Linears with GELU between+after."""
        mlp = _build_numeric_mlp(input_dim=5, hidden_dim=128, n_layers=2, activation="gelu")
        # Layout: Linear, GELU, Linear, GELU
        assert len(mlp) == 4
        assert isinstance(mlp[0], nn.Linear)
        assert isinstance(mlp[1], nn.GELU)
        assert isinstance(mlp[2], nn.Linear)
        assert isinstance(mlp[3], nn.GELU)
        # State_dict keys must be mlp.0.* and mlp.2.* (no others)
        keys = sorted(dict(mlp.named_parameters()).keys())
        assert keys == ["0.bias", "0.weight", "2.bias", "2.weight"]

    def test_n_layers_3(self):
        mlp = _build_numeric_mlp(input_dim=5, hidden_dim=128, n_layers=3, activation="gelu")
        # Linear, GELU, Linear, GELU, Linear, GELU
        assert len(mlp) == 6
        keys = sorted(dict(mlp.named_parameters()).keys())
        assert keys == ["0.bias", "0.weight", "2.bias", "2.weight", "4.bias", "4.weight"]

    def test_n_layers_1(self):
        mlp = _build_numeric_mlp(input_dim=5, hidden_dim=128, n_layers=1, activation="gelu")
        # Just Linear, GELU
        assert len(mlp) == 2
        keys = sorted(dict(mlp.named_parameters()).keys())
        assert keys == ["0.bias", "0.weight"]

    def test_activation_swap(self):
        for act_name, act_cls in [
            ("gelu", nn.GELU),
            ("relu", nn.ReLU),
            ("silu", nn.SiLU),
            ("tanh", nn.Tanh),
        ]:
            mlp = _build_numeric_mlp(input_dim=5, hidden_dim=64, n_layers=2, activation=act_name)
            assert isinstance(mlp[1], act_cls)
            assert isinstance(mlp[3], act_cls)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="numerical_activation"):
            _build_numeric_mlp(input_dim=5, hidden_dim=64, n_layers=2, activation="elu")

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="numerical_n_layers"):
            _build_numeric_mlp(input_dim=5, hidden_dim=64, n_layers=0, activation="gelu")


class TestNumericalEncoderKnobs:
    def test_default_state_dict_unchanged(self):
        """Default kwargs reproduce the pre-knob state_dict layout exactly."""
        enc = NumericalEncoder(input_dim=5, hidden_dim=128, output_dim=768)  # all defaults
        keys = sorted(enc.state_dict().keys())
        expected = sorted([
            "mlp.0.weight", "mlp.0.bias",
            "mlp.2.weight", "mlp.2.bias",
            "gamma_head.weight", "gamma_head.bias",
            "beta_head.weight", "beta_head.bias",
        ])
        assert keys == expected

    def test_n_layers_3_state_dict(self):
        enc = NumericalEncoder(input_dim=5, hidden_dim=64, output_dim=32, n_layers=3)
        keys = sorted(enc.state_dict().keys())
        assert "mlp.4.weight" in keys
        assert "mlp.4.bias" in keys

    def test_activation_silu_works(self):
        enc = NumericalEncoder(input_dim=5, hidden_dim=64, output_dim=32, activation="silu")
        x = torch.randn(3, 5)
        x[..., -1] = 1.0
        gamma, beta = enc(x)
        assert gamma.shape == (3, 32)
        assert torch.isfinite(gamma).all()


class TestAddCombiner:
    def test_output_shape(self):
        comb = AddCombiner(input_dim=5, hidden_dim=64, output_dim=32)
        x = torch.randn(8, 5)
        x[..., -1] = 1.0
        delta = comb(x)
        assert delta.shape == (8, 32)

    def test_zero_init_residual(self):
        """add_head zero-init means delta=0 at init regardless of input."""
        comb = AddCombiner(input_dim=5, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 5)
        x[..., -1] = 1.0
        delta = comb(x)
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_value_present_gating(self):
        """Non-numeric events (value_present=0) should produce zero delta."""
        comb = AddCombiner(input_dim=5, hidden_dim=64, output_dim=32)
        # Force add_head non-zero so we'd see a difference if gating failed
        with torch.no_grad():
            comb.add_head.weight.normal_()
            comb.add_head.bias.normal_()
        x = torch.randn(4, 5)
        x[..., -1] = 0.0  # value_present False
        delta = comb(x)
        assert torch.allclose(delta, torch.zeros_like(delta))


class TestConcatCombiner:
    def test_identity_init(self):
        """At init, output equals projected (numeric channel zero, identity on text)."""
        comb = ConcatCombiner(input_dim=5, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 5)
        x[..., -1] = 1.0
        projected = torch.randn(4, 32)
        out = comb(x, projected)
        assert torch.allclose(out, projected, atol=1e-5)

    def test_value_present_gating(self):
        """Numeric channel zeroed when value_present=0."""
        comb = ConcatCombiner(input_dim=5, hidden_dim=64, output_dim=32)
        # Make proj_head and concat_combine non-trivial
        with torch.no_grad():
            comb.proj_head.weight.normal_()
            comb.proj_head.bias.normal_()
            comb.concat_combine.weight.normal_()
            comb.concat_combine.bias.normal_()
        # With value_present=0, the result must equal Linear(cat([projected, 0]))
        x = torch.randn(4, 5)
        x[..., -1] = 0.0
        projected = torch.randn(4, 32)
        out = comb(x, projected)
        # Reference: same Linear applied to [projected, zeros]
        ref = comb.concat_combine(torch.cat([projected, torch.zeros_like(projected)], dim=-1))
        assert torch.allclose(out, ref, atol=1e-6)


class TestDualPathInputEncoderCombiners:
    def _make(self, combiner: str, num_dim: int = 5):
        text_emb = nn.Embedding(50, 128)
        text_emb.weight.requires_grad = False
        return DualPathInputEncoder(
            text_embedding=text_emb,
            hidden_size=64,
            numerical_input_dim=num_dim,
            numerical_hidden_dim=32,
            numerical_combiner=combiner,
        )

    def test_film_default_module_layout(self):
        enc = self._make("film")
        assert hasattr(enc, "numerical_encoder")
        assert not hasattr(enc, "numerical_add_combiner")
        assert not hasattr(enc, "numerical_concat_combiner")

    def test_add_module_layout(self):
        enc = self._make("add")
        assert hasattr(enc, "numerical_add_combiner")
        assert not hasattr(enc, "numerical_encoder")
        assert not hasattr(enc, "numerical_concat_combiner")

    def test_concat_module_layout(self):
        enc = self._make("concat_proj")
        assert hasattr(enc, "numerical_concat_combiner")
        assert not hasattr(enc, "numerical_encoder")
        assert not hasattr(enc, "numerical_add_combiner")

    def test_unknown_combiner_raises(self):
        with pytest.raises(ValueError, match="Unknown numerical_combiner"):
            text_emb = nn.Embedding(50, 128)
            DualPathInputEncoder(
                text_embedding=text_emb,
                hidden_size=64,
                numerical_combiner="weighted_sum",
            )

    @pytest.mark.parametrize("combiner", ["film", "add", "concat_proj"])
    def test_forward_runs_for_all_combiners(self, combiner):
        enc = self._make(combiner)
        ids = torch.randint(0, 50, (6,))
        feats = torch.randn(6, 5)
        feats[..., -1] = 1.0
        out = enc(ids, feats)
        assert out.shape == (6, 64)
        assert torch.isfinite(out).all()

    def test_film_default_matches_legacy_forward(self):
        """With combiner='film', the forward path is byte-identical to before
        the knob existed: gamma * projected + beta with gamma=1+head, beta=head."""
        enc = self._make("film")
        torch.manual_seed(123)
        ids = torch.randint(0, 50, (3,))
        feats = torch.randn(3, 5)
        feats[..., -1] = 1.0

        # Reference path: do it manually using the same submodules
        with torch.no_grad():
            text_emb = enc.text_embedding(ids)
            projected = enc.text_projection(text_emb)
            gamma, beta = enc.numerical_encoder(feats)
            ref = gamma * projected + beta
            actual = enc(ids, feats)
        assert torch.allclose(actual, ref)
