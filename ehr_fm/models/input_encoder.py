"""Dual-path input encoder for embedding-mode EHRFM (Arm A1).

Replaces discrete nn.Embedding with:
- Frozen text embedding lookup → learned projection
- Optional combiner over numeric event features. The default ("film") is the
  classic FiLM-style modulation. Alternative combiners ("add", "concat_proj")
  are opt-in and produce different state_dict keys; they do not affect
  checkpoints saved with the default combiner.
"""

from __future__ import annotations

import torch
from torch import nn

# Map config strings to torch activation classes. All are stateless — fresh
# instances are created per-position in nn.Sequential.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _build_numeric_mlp(input_dim: int, hidden_dim: int, n_layers: int, activation: str) -> nn.Sequential:
    """Build the shared MLP used by every numeric combiner.

    Structure: Linear, act, Linear, act, ..., Linear, act
    Trailing activation is intentional and matches the pre-knob default. With
    ``n_layers=2, activation="gelu"`` (the defaults), the produced module has
    state_dict keys ``mlp.0.{weight,bias}`` and ``mlp.2.{weight,bias}`` —
    byte-compatible with checkpoints saved before these knobs were added.
    """
    if n_layers < 1:
        raise ValueError(f"numerical_n_layers must be >= 1, got {n_layers}")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"numerical_activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
    act_cls = _ACTIVATIONS[activation]

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_cls()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_cls())
    return nn.Sequential(*layers)


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

    Produces gamma and beta modulation vectors from the numeric feature
    vector (last position is the value_present flag).

    Uses residual gamma parameterization (gamma = 1 + head output) so that
    weight decay pushes toward identity rather than away from it.

    Hard gate on value_present: non-numeric events bypass FiLM entirely.

    The ``n_layers`` and ``activation`` kwargs were added later. Their
    defaults reproduce the original 2-Linear-with-GELU MLP exactly, so
    existing checkpoints (saved without these fields in config.json) load
    with no missing/unexpected state_dict keys.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 768,
        n_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.mlp = _build_numeric_mlp(input_dim, hidden_dim, n_layers, activation)
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
        mask = value_present > 0.5  # [*, 1] broadcasts over output_dim
        gamma = torch.where(mask, gamma, torch.ones_like(gamma))
        beta = torch.where(mask, beta, torch.zeros_like(beta))

        return gamma, beta


class AddCombiner(nn.Module):
    """Additive numeric combiner.

    Projects the numeric features through an MLP and an output head, then
    adds the result to the text-projected hidden state. Zero-init head
    means non-numeric events (and untrained init) start as identity.

    Forward returns the residual delta (not the combined output) so the
    caller composes ``projected + delta``. Hard-gates on value_present.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 768,
        n_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.mlp = _build_numeric_mlp(input_dim, hidden_dim, n_layers, activation)
        self.add_head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.add_head.weight)
        nn.init.zeros_(self.add_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value_present = x[..., -1:]
        h = self.mlp(x)
        delta = self.add_head(h)
        mask = value_present > 0.5
        return torch.where(mask, delta, torch.zeros_like(delta))


class ConcatCombiner(nn.Module):
    """Concat-then-project numeric combiner.

    Projects numeric features to ``output_dim`` (gated by value_present),
    concatenates with the text-projected hidden state, and projects the
    concatenation back to ``output_dim``. Identity-initialized so the
    untrained model is equivalent to text-only at init.

    Forward takes both ``x`` (numeric features) and ``projected`` (text
    path) and returns the combined hidden state.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 768,
        n_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.mlp = _build_numeric_mlp(input_dim, hidden_dim, n_layers, activation)
        self.proj_head = nn.Linear(hidden_dim, output_dim)
        self.concat_combine = nn.Linear(output_dim * 2, output_dim)

        # Identity init: the [projected, numeric] concat should pass projected
        # through unchanged at start. Zero numeric channel + identity on
        # projected channel + zero bias.
        with torch.no_grad():
            W = self.concat_combine.weight  # shape (output_dim, 2*output_dim)
            W.zero_()
            W[:, :output_dim] = torch.eye(output_dim)
            self.concat_combine.bias.zero_()

    def forward(self, x: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
        value_present = x[..., -1:]
        h = self.mlp(x)
        num = self.proj_head(h)
        mask = value_present > 0.5
        num = torch.where(mask, num, torch.zeros_like(num))
        return self.concat_combine(torch.cat([projected, num], dim=-1))


class DualPathInputEncoder(nn.Module):
    """Dual-path input encoder: frozen text embedding + optional numeric combiner.

    Replaces nn.Embedding in embedding mode. Output shape matches (seq_len, hidden_size).

    The numeric path's combiner is selected by ``numerical_combiner``:
      - ``"film"`` (default): FiLM modulation via :class:`NumericalEncoder`.
        State_dict keys: ``numerical_encoder.{mlp.*,gamma_head.*,beta_head.*}``.
      - ``"add"``: residual add via :class:`AddCombiner`.
        State_dict keys: ``numerical_add_combiner.{mlp.*,add_head.*}``.
      - ``"concat_proj"``: concat-then-project via :class:`ConcatCombiner`.
        State_dict keys: ``numerical_concat_combiner.{mlp.*,proj_head.*,concat_combine.*}``.

    Existing checkpoints (saved before the combiner knob was added) default
    to ``"film"`` and load unchanged.
    """

    def __init__(
        self,
        text_embedding: nn.Embedding,
        hidden_size: int,
        use_numerical_path: bool = True,
        numerical_input_dim: int = 5,
        numerical_hidden_dim: int = 128,
        numerical_n_layers: int = 2,
        numerical_activation: str = "gelu",
        numerical_combiner: str = "film",
    ):
        super().__init__()
        self.text_embedding = text_embedding
        embedding_dim = text_embedding.embedding_dim
        self.text_projection = TextProjection(embedding_dim, hidden_size)

        self.use_numerical_path = use_numerical_path
        self.numerical_combiner = numerical_combiner

        if use_numerical_path:
            kwargs = dict(
                input_dim=numerical_input_dim,
                hidden_dim=numerical_hidden_dim,
                output_dim=hidden_size,
                n_layers=numerical_n_layers,
                activation=numerical_activation,
            )
            if numerical_combiner == "film":
                self.numerical_encoder = NumericalEncoder(**kwargs)
            elif numerical_combiner == "add":
                self.numerical_add_combiner = AddCombiner(**kwargs)
            elif numerical_combiner == "concat_proj":
                self.numerical_concat_combiner = ConcatCombiner(**kwargs)
            else:
                raise ValueError(
                    f"Unknown numerical_combiner: {numerical_combiner!r}. "
                    f"Expected one of: 'film', 'add', 'concat_proj'."
                )

    def forward(
        self, embedding_text_ids: torch.Tensor, numeric_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            text_emb = self.text_embedding(embedding_text_ids)

        projected = self.text_projection(text_emb)

        if not self.use_numerical_path or numeric_features is None:
            return projected

        if self.numerical_combiner == "film":
            gamma, beta = self.numerical_encoder(numeric_features)
            return gamma * projected + beta
        if self.numerical_combiner == "add":
            return projected + self.numerical_add_combiner(numeric_features)
        if self.numerical_combiner == "concat_proj":
            return self.numerical_concat_combiner(numeric_features, projected)
        # Should be unreachable thanks to the __init__ check.
        raise RuntimeError(f"Unhandled numerical_combiner: {self.numerical_combiner!r}")
