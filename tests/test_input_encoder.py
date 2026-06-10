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
        self,
        num_embeddings=100,
        embedding_dim=128,
        hidden_size=64,
        use_numerical=True,
        numerical_input_dim=5,
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
        from ehr_fm.data import packed_ehr_collate

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
        """ref_range_priority: 4-dim numeric features collate by concatenation, values intact."""
        from ehr_fm.data import packed_ehr_collate

        f1 = torch.tensor([[0.5, 1.0, 0.0, 1.0], [0.2, 0.0, 1.0, 1.0]])
        f2 = torch.tensor([[1.5, 1.0, 0.0, 1.0]])
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
                "numeric_features": f1,
            },
            {
                "input_ids": torch.tensor([3]),
                "labels": torch.tensor([-100]),
                "age": torch.tensor([0.0]),
                "age_normalized": torch.tensor([0.0]),
                "length": 1,
                "patient_id": 2,
                "index_time": 2000.0,
                "embedding_text_ids": torch.tensor([30]),
                "numeric_features": f2,
            },
        ]
        collated = packed_ehr_collate(batch)
        assert collated["numeric_features"].shape == (3, 4)
        assert torch.equal(collated["numeric_features"], torch.cat([f1, f2], dim=0))

    def test_collate_with_dim15_numeric_features(self):
        """fourier_ref_range_priority: 15-dim numeric features collate, values intact."""
        from ehr_fm.data import packed_ehr_collate

        feats = torch.arange(2 * 15, dtype=torch.float32).reshape(2, 15)
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
                "numeric_features": feats,
            },
        ]
        collated = packed_ehr_collate(batch)
        assert collated["numeric_features"].shape == (2, 15)
        assert torch.equal(collated["numeric_features"], feats)

    def test_collate_without_embedding_fields(self):
        """Backward compat: batch without embedding fields still works."""
        from ehr_fm.data import packed_ehr_collate

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
