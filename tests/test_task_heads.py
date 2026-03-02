import pytest
import torch

from ehr_fm.models.task_heads import SequenceClassificationHead, make_task_head


class TestSequenceClassificationHead:
    def test_output_shape(self):
        head = SequenceClassificationHead(hidden_size=32, n_classes=10)
        features = torch.randn(4, 32)
        labels = torch.randint(0, 10, (4,))
        loss, result = head(features, {"labels": labels})
        assert loss.shape == ()
        assert result["logits"] is None

    def test_return_logits_true(self):
        head = SequenceClassificationHead(hidden_size=32, n_classes=10)
        features = torch.randn(4, 32)
        labels = torch.randint(0, 10, (4,))
        loss, result = head(features, {"labels": labels}, return_logits=True)
        assert result["logits"] is not None
        assert result["logits"].shape == (4, 10)

    def test_return_logits_false(self):
        head = SequenceClassificationHead(hidden_size=32, n_classes=10)
        features = torch.randn(4, 32)
        labels = torch.randint(0, 10, (4,))
        _, result = head(features, {"labels": labels}, return_logits=False)
        assert result["logits"] is None

    def test_ignore_index(self):
        head = SequenceClassificationHead(hidden_size=32, n_classes=10)
        features = torch.randn(4, 32)
        labels = torch.tensor([0, -100, 3, -100])
        loss, _ = head(features, {"labels": labels})
        assert loss.isfinite()

    def test_default_n_classes(self):
        head = SequenceClassificationHead(hidden_size=64)
        assert head.final_layer.out_features == 8_192


class TestMakeTaskHead:
    def test_sequence_classification(self):
        head = make_task_head("sequence_classification", hidden_size=32, n_classes=5)
        assert isinstance(head, SequenceClassificationHead)
        assert head.final_layer.out_features == 5

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            make_task_head("nonexistent_type", hidden_size=32)

    def test_default_type(self):
        head = make_task_head(hidden_size=16)
        assert isinstance(head, SequenceClassificationHead)
