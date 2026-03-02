from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int = 8_192):
        super().__init__()
        self.final_layer = nn.Linear(hidden_size, n_classes)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], *, return_logits=False):
        logits = self.final_layer(features)
        loss = F.cross_entropy(logits, batch["labels"], ignore_index=-100)
        return loss, {"logits": logits if return_logits else None}


def make_task_head(task_type: str = "sequence_classification", *args, **kwargs) -> nn.Module:
    if task_type == "sequence_classification":
        return SequenceClassificationHead(*args, **kwargs)
    raise ValueError(f"Unknown task type: {task_type}")
