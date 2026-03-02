"""
Unit tests for representation computation logic.

Tests cover:
1. Last feature selection via cumsum(patient_lengths) - 1
2. Dtype handling for representations (bf16/fp16)
3. Timestep-based patient length handling
4. Patient ID alignment with representations

All tests focus on the cumsum indexing logic used in compute_representations.
"""

import torch

# -----------------------------------------------------------------------------
# Test 4.1: Last Feature Selection
# -----------------------------------------------------------------------------


class TestLastFeatureSelection:
    """Test that last features are correctly selected via cumsum."""

    def test_cumsum_indexing_basic(self):
        """Test that cumsum - 1 correctly selects last token per patient."""
        # Create batch with known patient lengths
        patient_lengths = torch.tensor([3, 5, 2], dtype=torch.int32)
        total_tokens = patient_lengths.sum().item()

        # Simulate hidden states (mock representation vectors)
        embedding_dim = 64
        hidden_states = torch.randn(total_tokens, embedding_dim)

        # Select last features using cumsum (this is the key logic we're testing)
        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Should have one representation per patient
        assert patient_representations.shape[0] == len(patient_lengths)
        assert patient_representations.shape[1] == embedding_dim

        # Verify indices are correct
        assert end_indices[0].item() == 2  # Patient 1: positions 0,1,2 -> last is 2
        assert end_indices[1].item() == 7  # Patient 2: positions 3-7 -> last is 7
        assert end_indices[2].item() == 9  # Patient 3: positions 8,9 -> last is 9

        # Verify we're selecting the correct vectors
        assert torch.equal(patient_representations[0], hidden_states[2])
        assert torch.equal(patient_representations[1], hidden_states[7])
        assert torch.equal(patient_representations[2], hidden_states[9])

    def test_cumsum_with_varying_lengths(self):
        """Test cumsum indexing with varied patient lengths."""
        # More realistic distribution
        patient_lengths = torch.tensor([1, 10, 5, 3, 20], dtype=torch.int32)
        total_tokens = patient_lengths.sum().item()

        hidden_states = torch.randn(total_tokens, 64)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Check shape
        assert patient_representations.shape == (5, 64)

        # Check indices
        expected_end_indices = [0, 10, 15, 18, 38]
        assert torch.equal(end_indices, torch.tensor(expected_end_indices, dtype=torch.int32))

    def test_cumsum_single_patient(self):
        """Test with single patient."""
        patient_lengths = torch.tensor([10], dtype=torch.int32)
        hidden_states = torch.randn(10, 64)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Single patient, last position should be index 9
        assert end_indices[0].item() == 9
        assert patient_representations.shape == (1, 64)

    def test_cumsum_with_single_token_patients(self):
        """Test with patients having single tokens."""
        # Edge case: some patients have only 1 token
        patient_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        hidden_states = torch.randn(3, 64)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Each patient's only token is their representation
        assert torch.equal(end_indices, torch.tensor([0, 1, 2], dtype=torch.int32))
        assert patient_representations.shape == (3, 64)


# -----------------------------------------------------------------------------
# Test 4.2: Dtype Handling
# -----------------------------------------------------------------------------


class TestDtypeHandling:
    """Test dtype handling in representation selection."""

    def test_float32_representations(self):
        """Test with standard float32 representations."""
        patient_lengths = torch.tensor([3, 5], dtype=torch.int32)
        total_tokens = patient_lengths.sum().item()

        hidden_states = torch.randn(total_tokens, 64, dtype=torch.float32)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Should work without errors
        assert patient_representations.shape == (2, 64)
        assert patient_representations.dtype == torch.float32
        assert not torch.any(torch.isnan(patient_representations))

    def test_bfloat16_representations(self):
        """Test with bfloat16 representations."""
        patient_lengths = torch.tensor([3, 5], dtype=torch.int32)
        total_tokens = patient_lengths.sum().item()

        hidden_states = torch.randn(total_tokens, 64, dtype=torch.float32).to(torch.bfloat16)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Should work with bf16
        assert patient_representations.dtype == torch.bfloat16
        assert patient_representations.shape == (2, 64)


# -----------------------------------------------------------------------------
# Test 4.3: Timestep-based Patient Lengths
# -----------------------------------------------------------------------------


class TestTimestepRepresentations:
    """Test representation computation when patient_lengths refers to timesteps."""

    def test_timestep_last_feature_selection(self):
        """Test last feature selection when patient_lengths are timestep counts.

        When patient_lengths refers to number of timesteps (not tokens),
        the cumsum logic remains the same.
        """
        # Patient 1: 3 timesteps
        # Patient 2: 2 timesteps
        # Total: 5 timesteps in sequence
        patient_lengths = torch.tensor([3, 2], dtype=torch.int32)

        # Simulate hidden states at timestep level (5 timesteps total)
        hidden_states = torch.randn(5, 64)

        # cumsum(patient_lengths) - 1 gives last timestep index per patient
        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        assert end_indices[0].item() == 2  # Patient 1: last timestep index is 2
        assert end_indices[1].item() == 4  # Patient 2: last timestep index is 4
        assert patient_representations.shape == (2, 64)


# -----------------------------------------------------------------------------
# Test 4.4: Patient ID Alignment
# -----------------------------------------------------------------------------


class TestPatientIdAlignment:
    """Test that patient IDs align correctly with representations."""

    def test_patient_id_order_preservation(self):
        """Test that patient IDs remain in correct order."""
        # Simulate representation computation tracking
        patient_ids_batch1 = [1001, 1002, 1003]
        patient_ids_batch2 = [2001, 2002]
        patient_ids_batch3 = [3001]

        all_patient_ids = []
        all_patient_ids.extend(patient_ids_batch1)
        all_patient_ids.extend(patient_ids_batch2)
        all_patient_ids.extend(patient_ids_batch3)

        # Should maintain order
        expected = [1001, 1002, 1003, 2001, 2002, 3001]
        assert all_patient_ids == expected

    def test_patient_id_matches_representation_count(self):
        """Test that number of patient IDs matches number of representations."""
        # Mock representations
        repr_batch1 = torch.randn(3, 64)  # 3 patients
        repr_batch2 = torch.randn(2, 64)  # 2 patients

        all_representations = [repr_batch1, repr_batch2]
        flattened = torch.cat(all_representations, dim=0)

        # Patient IDs
        all_patient_ids = [1, 2, 3, 4, 5]

        # Should match
        assert flattened.shape[0] == len(all_patient_ids)


# -----------------------------------------------------------------------------
# Test 4.5: Edge Cases
# -----------------------------------------------------------------------------


class TestRepresentationEdgeCases:
    """Test edge cases in representation computation."""

    def test_very_long_patient(self):
        """Test with very long patient sequence."""
        patient_lengths = torch.tensor([500], dtype=torch.int32)
        hidden_states = torch.randn(500, 32)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Should handle long sequence
        assert end_indices[0].item() == 499
        assert patient_representations.shape == (1, 32)
        assert not torch.any(torch.isnan(patient_representations))

    def test_many_short_patients(self):
        """Test with many patients with short sequences."""
        # 100 patients with 1-5 tokens each
        patient_lengths = torch.randint(1, 6, (100,), dtype=torch.int32)
        total_tokens = patient_lengths.sum().item()

        hidden_states = torch.randn(total_tokens, 32)

        end_indices = torch.cumsum(patient_lengths, dim=0) - 1
        patient_representations = hidden_states[end_indices]

        # Should handle many patients
        assert patient_representations.shape == (100, 32)
        assert not torch.any(torch.isnan(patient_representations))

        # Verify all end indices are valid
        assert torch.all(end_indices >= 0)
        assert torch.all(end_indices < total_tokens)
