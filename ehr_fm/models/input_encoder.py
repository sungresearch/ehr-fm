"""Dual-path input encoder for embedding-mode EHRFM (Arm A1).

Replaces discrete nn.Embedding with:
- Frozen text embedding lookup → learned projection
- Optional FiLM-based numerical encoder for numeric event features
"""

from __future__ import annotations

import torch
from torch import nn


class TextProjection(nn.Module):
    """Projects frozen text embeddings to transformer hidden size.

    2-layer MLP: Linear(embedding_dim, hidden_size) → GELU → Linear(hidden_size, hidden_size)
    """

    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NumericalEncoder(nn.Module):
    """FiLM-style encoder for numerical features.

    Produces gamma and beta modulation vectors from a 5-dim feature vector:
    [value_log_zscore, quantile, ref_range_position, ref_range_available, value_present]

    Uses residual gamma parameterization (gamma = 1 + head output) so that
    weight decay pushes toward identity rather than away from it.

    Hard gate on value_present: non-numeric events bypass FiLM entirely.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, output_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, output_dim)
        self.beta_head = nn.Linear(hidden_dim, output_dim)

        # Residual parameterization: gamma = 1 + head(h), so head should
        # output 0 at init. Zero weights + zero bias achieves this.
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value_present = x[..., -1:]  # [*, 1]

        h = self.mlp(x)
        gamma = 1.0 + self.gamma_head(h)
        beta = self.beta_head(h)

        # Hard gate: identity pass-through for non-numeric events
        mask = (value_present > 0.5)  # [*, 1] broadcasts over output_dim
        gamma = torch.where(mask, gamma, torch.ones_like(gamma))
        beta = torch.where(mask, beta, torch.zeros_like(beta))

        return gamma, beta


class DualPathInputEncoder(nn.Module):
    """Dual-path input encoder: frozen text embedding + optional numerical FiLM.

    Replaces nn.Embedding in embedding mode. Output shape matches (seq_len, hidden_size).
    """

    def __init__(
        self,
        text_embedding: nn.Embedding,
        hidden_size: int,
        use_numerical_path: bool = True,
        numerical_input_dim: int = 5,
        numerical_hidden_dim: int = 128,
    ):
        super().__init__()
        self.text_embedding = text_embedding
        embedding_dim = text_embedding.embedding_dim
        self.text_projection = TextProjection(embedding_dim, hidden_size)

        self.use_numerical_path = use_numerical_path
        if use_numerical_path:
            self.numerical_encoder = NumericalEncoder(
                input_dim=numerical_input_dim,
                hidden_dim=numerical_hidden_dim,
                output_dim=hidden_size,
            )

    def forward(
        self, embedding_text_ids: torch.Tensor, numeric_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            text_emb = self.text_embedding(embedding_text_ids)

        projected = self.text_projection(text_emb)

        if self.use_numerical_path and numeric_features is not None:
            gamma, beta = self.numerical_encoder(numeric_features)
            return gamma * projected + beta

        return projected
