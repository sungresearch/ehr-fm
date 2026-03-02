import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from ehr_fm.trainer import sample_indices


def _make_mock_dataset_and_sampler(n_items: int, lengths: list[int], tokens_per_batch: int):
    """Build mock objects that expose the interfaces sample_indices needs."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=n_items)

    sampler = SimpleNamespace(lengths=lengths, tokens_per_batch=tokens_per_batch, min_patients=1)
    return dataset, sampler


class TestSampleIndices:
    def test_collects_up_to_budget(self):
        lengths = [100, 100, 100, 100, 100]
        ds, sampler = _make_mock_dataset_and_sampler(5, lengths, tokens_per_batch=200)
        random.seed(42)
        indices = sample_indices(ds, sampler, k_batches=1, ensure_random=False)
        total = sum(lengths[i] for i in indices)
        assert total <= 200

    def test_k_batches_limits_output(self):
        lengths = [50] * 20
        ds, sampler = _make_mock_dataset_and_sampler(20, lengths, tokens_per_batch=100)
        random.seed(42)
        indices = sample_indices(ds, sampler, k_batches=2, ensure_random=False)
        total = sum(lengths[i] for i in indices)
        assert total < 2 * 100

    def test_ensure_random_varies(self):
        lengths = [10] * 100
        ds, sampler = _make_mock_dataset_and_sampler(100, lengths, tokens_per_batch=50)
        results = set()
        for _ in range(5):
            indices = sample_indices(ds, sampler, k_batches=2, ensure_random=True)
            results.add(tuple(sorted(indices)))
        assert len(results) > 1

    def test_ensure_random_false_reproducible(self):
        lengths = [10] * 50
        ds, sampler = _make_mock_dataset_and_sampler(50, lengths, tokens_per_batch=50)
        random.seed(123)
        r1 = sample_indices(ds, sampler, k_batches=2, ensure_random=False)
        random.seed(123)
        r2 = sample_indices(ds, sampler, k_batches=2, ensure_random=False)
        assert r1 == r2

    def test_k_batches_zero_returns_empty(self):
        lengths = [10] * 10
        ds, sampler = _make_mock_dataset_and_sampler(10, lengths, tokens_per_batch=50)
        indices = sample_indices(ds, sampler, k_batches=0, ensure_random=False)
        assert indices == []

    def test_single_large_item(self):
        lengths = [1000]
        ds, sampler = _make_mock_dataset_and_sampler(1, lengths, tokens_per_batch=100)
        random.seed(0)
        indices = sample_indices(ds, sampler, k_batches=1, ensure_random=False)
        assert len(indices) == 0


class TestComputeLoss:
    def test_return_outputs_false(self):
        from ehr_fm.trainer import FMTrainer

        loss_val = torch.tensor(2.5)
        mock_model = MagicMock(return_value=(loss_val, {"logits": None}))
        inputs = {"input_ids": torch.zeros(4)}

        trainer = FMTrainer.__new__(FMTrainer)
        result = trainer.compute_loss(mock_model, inputs, return_outputs=False)
        assert torch.equal(result, loss_val)

    def test_return_outputs_true(self):
        from ehr_fm.trainer import FMTrainer

        loss_val = torch.tensor(1.0)
        outputs = {"logits": torch.randn(4, 10)}
        mock_model = MagicMock(return_value=(loss_val, outputs))
        inputs = {"input_ids": torch.zeros(4)}

        trainer = FMTrainer.__new__(FMTrainer)
        result = trainer.compute_loss(mock_model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], loss_val)
        assert result[1] is outputs
