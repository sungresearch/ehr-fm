"""Tests for separate RoPE configurations for sparse vs global attention layers."""

import pytest
import torch
import torch.testing

from ehr_fm.models.transformer import (
    DenseTransformer,
    DenseTransformerConfig,
    _fixed_pos_embedding,
)


def test_fixed_pos_embedding_base_parameter():
    """Test that _fixed_pos_embedding respects the base parameter."""
    ages = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float32)
    dim = 8
    dtype = torch.float32

    # Test with different bases
    sin_cos_base_100 = _fixed_pos_embedding(ages, dim, dtype, base=100.0)
    sin_cos_base_10000 = _fixed_pos_embedding(ages, dim, dtype, base=10000.0)

    # Embeddings with different bases should be different
    assert not torch.allclose(sin_cos_base_100[0], sin_cos_base_10000[0], rtol=1e-4)
    assert not torch.allclose(sin_cos_base_100[1], sin_cos_base_10000[1], rtol=1e-4)

    # Default base should match explicit base=10000
    sin_cos_default = _fixed_pos_embedding(ages, dim, dtype)
    assert torch.allclose(sin_cos_default[0], sin_cos_base_10000[0], rtol=1e-6)
    assert torch.allclose(sin_cos_default[1], sin_cos_base_10000[1], rtol=1e-6)


def test_separate_rope_config_disabled():
    """Test that separate_rope_by_attention=False uses rope_base_global for all layers."""
    config = DenseTransformerConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        n_heads=4,
        n_layers=4,
        attention_width=4,
        alternating_dense_layers=True,
        dense_every_n_layers=2,
        rope_base_sparse=100.0,
        rope_base_global=10000.0,
        separate_rope_by_attention=False,  # Disabled
    )
    transformer = DenseTransformer(config)
    transformer.eval()

    # Create test batch
    batch = {
        "input_ids": torch.randint(0, 128, (10,), dtype=torch.long),
        "ages": torch.arange(10, dtype=torch.float32),
        "normalized_ages": torch.zeros(10, dtype=torch.float32),
        "patient_lengths": torch.tensor([10], dtype=torch.int32),
    }

    # Forward should work without errors
    with torch.no_grad():
        output = transformer(batch)

    assert output.shape == (10, 32)  # (seq_len, hidden_size)


def test_separate_rope_config_enabled():
    """Test that separate_rope_by_attention=True uses different RoPE bases per layer type."""
    config = DenseTransformerConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        n_heads=4,
        n_layers=4,
        attention_width=4,
        alternating_dense_layers=True,
        dense_every_n_layers=2,
        rope_base_sparse=100.0,
        rope_base_global=10000.0,
        separate_rope_by_attention=True,  # Enabled
    )
    transformer = DenseTransformer(config)
    transformer.eval()

    # Create test batch
    batch = {
        "input_ids": torch.randint(0, 128, (10,), dtype=torch.long),
        "ages": torch.arange(10, dtype=torch.float32),
        "normalized_ages": torch.zeros(10, dtype=torch.float32),
        "patient_lengths": torch.tensor([10], dtype=torch.int32),
    }

    # Forward should work without errors
    with torch.no_grad():
        output = transformer(batch)

    assert output.shape == (10, 32)  # (seq_len, hidden_size)


def test_separate_rope_produces_different_outputs():
    """Test that separate RoPE configs produce different outputs than unified config."""
    torch.manual_seed(42)

    base_config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 48,
        "n_heads": 4,
        "n_layers": 4,
        "attention_width": 4,
        "alternating_dense_layers": True,
        "dense_every_n_layers": 2,
        "rope_base_sparse": 100.0,
        "rope_base_global": 10000.0,
    }

    # Model with separate RoPE disabled (uses rope_base_global for all)
    config_unified = DenseTransformerConfig(**base_config, separate_rope_by_attention=False)
    transformer_unified = DenseTransformer(config_unified)
    transformer_unified.eval()

    # Model with separate RoPE enabled
    config_separate = DenseTransformerConfig(**base_config, separate_rope_by_attention=True)
    transformer_separate = DenseTransformer(config_separate)
    transformer_separate.eval()

    # Copy weights to ensure differences are only from RoPE
    transformer_separate.load_state_dict(transformer_unified.state_dict())

    # Create test batch
    batch = {
        "input_ids": torch.randint(0, 128, (10,), dtype=torch.long),
        "ages": torch.arange(10, dtype=torch.float32),
        "normalized_ages": torch.zeros(10, dtype=torch.float32),
        "patient_lengths": torch.tensor([10], dtype=torch.int32),
    }

    # Forward pass
    with torch.no_grad():
        output_unified = transformer_unified(batch)
        output_separate = transformer_separate(batch)

    # Outputs should be different due to different RoPE configurations
    assert output_unified.shape == output_separate.shape
    assert not torch.allclose(output_unified, output_separate, rtol=1e-3, atol=1e-5)


def test_backward_compatibility():
    """Test that models with default config (separate_rope_by_attention=False) maintain backward compatibility."""
    # Old-style config without new parameters (should use defaults)
    config_old_style = DenseTransformerConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        n_heads=4,
        n_layers=4,
        attention_width=4,
        alternating_dense_layers=True,
        dense_every_n_layers=2,
    )

    # Verify defaults
    assert config_old_style.rope_base_sparse == 100.0
    assert config_old_style.rope_base_global == 10000.0
    assert config_old_style.separate_rope_by_attention is False

    transformer = DenseTransformer(config_old_style)
    transformer.eval()

    # Should work exactly as before
    batch = {
        "input_ids": torch.randint(0, 128, (10,), dtype=torch.long),
        "ages": torch.arange(10, dtype=torch.float32),
        "normalized_ages": torch.zeros(10, dtype=torch.float32),
        "patient_lengths": torch.tensor([10], dtype=torch.int32),
    }

    with torch.no_grad():
        output = transformer(batch)

    assert output.shape == (10, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
