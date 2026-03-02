"""
EHRFM dense transformer model for EHR foundation model pretraining adapted from CLMBR
https://github.com/som-shahlab/femr/blob/main/src/femr/models/transformer.py

Key differences:
- Use alternating global and local causal attention layers
- Linear layer bias is not used by default
- GELU is enabled by default
- First encoder layer norm is removed to reduce duplicated normalization at the start
- Use 1.5 * hidden_size for intermediate size
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
import transformers
import xformers.ops
from torch import nn

from .config import DenseTransformerConfig, EHRFMConfig
from .task_heads import make_task_head
from .utils import RMSNorm, memory_efficient_attention_wrapper


# -----------------------------------------------------------------------------
# EHRFM Collate function
# -----------------------------------------------------------------------------
def packed_ehr_collate(batch):
    """
    Combine variable-length windows into the flattened EHRFM layout.

    Output:
        { input_ids, ages, normalized_ages, patient_lengths, label_indices, patient_ids, index_times, task: {labels} }
    """
    lens = [ex["length"] for ex in batch]
    ages = torch.cat([ex["age"] for ex in batch])
    anorm = torch.cat([ex["age_normalized"] for ex in batch])
    lbls = torch.cat([ex["labels"] for ex in batch])
    pids = [ex["patient_id"] for ex in batch]
    idx_t = [ex["index_time"] for ex in batch]

    device = ages.device
    patient_lengths = torch.tensor(lens, dtype=torch.int32, device=device)
    label_indices = torch.nonzero(lbls != -100, as_tuple=False)[:, 0].long()
    patient_ids = torch.tensor(pids, dtype=torch.int64, device=device)
    index_times = torch.tensor(idx_t, dtype=torch.int64, device=device)

    toks = torch.cat([ex["input_ids"] for ex in batch])

    result = {
        "input_ids": toks,
        "ages": ages,
        "normalized_ages": anorm,
        "patient_lengths": patient_lengths,
        "label_indices": label_indices,
        "patient_ids": patient_ids,
        "index_times": index_times,
        "task": {
            "labels": lbls[label_indices],
        },
    }

    return result


# -----------------------------------------------------------------------------
# Rotary positional helpers
# -----------------------------------------------------------------------------
def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    x1, x2 = flat[:, ::2], flat[:, 1::2]
    result = torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    return result


def _fixed_pos_embedding(ages: torch.Tensor, dim: int, dtype: torch.dtype, base: float = 10000.0):
    """Sin‑cos positional embedding on *age in days* (scalar per token).

    Args:
        ages: Age in days per token, shape (num_tokens,)
        dim: Embedding dimension (typically head_dim)
        dtype: Output tensor dtype
        base: Base value for frequency computation. With linspace(0,2), effective theta = base^2.
              - base=100 → effective theta ~10K
              - base=10000 → effective theta ~100M

    Returns:
        Tuple of (sin, cos) tensors, each of shape (num_tokens, 1, dim)
    """
    inv_freq = 1.0 / (base ** (torch.linspace(0, 2, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    ages = ages.reshape(ages.shape[0], 1)
    t = inv_freq * ages
    sin, cos = torch.sin(t), torch.cos(t)
    final_shape = (ages.shape[0], 1, dim)
    sin = torch.stack((sin, sin), dim=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), dim=-1).reshape(final_shape).type(dtype)
    return sin, cos


def _apply_rotary(x: torch.Tensor, sincos: tuple[torch.Tensor, torch.Tensor]):
    sin, cos = sincos
    sin, cos = sin.float(), cos.float()
    if len(sin.shape) != len(x.shape):
        new_shape = (1,) + sin.shape
        sin, cos = sin.reshape(new_shape), cos.reshape(new_shape)

    x32 = x.float()
    y32 = (x32 * cos) + (_rotate_every_two(x32) * sin)
    return y32.to(x.dtype)


# -----------------------------------------------------------------------------
# Decoder layer (attention + FFN)
# -----------------------------------------------------------------------------
class DenseTransformerDecoderLayer(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.config = config

        self.norm = RMSNorm(config.hidden_size)
        hidden_mult = 2 if config.hidden_act == "swiglu" else 1

        self.input_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3 + hidden_mult * config.intermediate_size,
            bias=config.use_bias,
        )
        self.output_proj = nn.Linear(
            config.hidden_size + config.intermediate_size, config.hidden_size, bias=config.use_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        normed_ages: torch.Tensor,
        pos_embed: tuple[torch.Tensor, torch.Tensor],
        attn_bias: xformers.ops.AttentionBias,
    ) -> torch.Tensor:
        x = self.norm(x)

        if self.config.use_normed_ages:
            x[:, -2] = normed_ages.to(dtype=x.dtype)
            x[:, -1] = (normed_ages**2).to(dtype=x.dtype)

        transformed = self.input_proj(x)
        ff, qkv = (
            transformed[:, : -self.config.hidden_size * 3],
            transformed[:, -self.config.hidden_size * 3 :],
        )

        head_dim = self.config.hidden_size // self.config.n_heads
        qkv = qkv.view(x.shape[0], 3, self.config.n_heads, head_dim)
        q = _apply_rotary(qkv[:, 0], pos_embed)
        k = _apply_rotary(qkv[:, 1], pos_embed)
        v = qkv[:, 2]

        attn = memory_efficient_attention_wrapper(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_bias=attn_bias
        ).reshape(x.shape)

        if self.config.hidden_act == "gelu":
            ff = F.gelu(ff)
        elif self.config.hidden_act == "swiglu":
            x1, x2 = ff.chunk(2, dim=-1)
            ff = F.silu(x1) * x2

        combined = torch.cat((attn, ff), dim=-1)
        return self.output_proj(combined)


# -----------------------------------------------------------------------------
# Transformer backbone
# -----------------------------------------------------------------------------
class DenseTransformer(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = RMSNorm(config.hidden_size)
        self.out_norm = RMSNorm(config.hidden_size)

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([DenseTransformerDecoderLayer(config) for _ in range(config.n_layers)])

        if config.remove_first_block_norm:
            self.layers[0].norm = nn.Identity()

        # pre-compute which layers should use dense attention
        interval = config.dense_every_n_layers if config.alternating_dense_layers else None
        if interval is None:
            self._dense_mask = [False] * config.n_layers
        else:
            self._dense_mask = [(i % interval == 0) for i in range(config.n_layers)]

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        x = self.embed(batch["input_ids"])

        x = self.in_norm(x)
        head_dim = self.config.hidden_size // self.config.n_heads

        if self.config.separate_rope_by_attention:
            pos_embed_sparse = _fixed_pos_embedding(
                batch["ages"], head_dim, x.dtype, base=self.config.rope_base_sparse
            )
            pos_embed_global = _fixed_pos_embedding(
                batch["ages"], head_dim, x.dtype, base=self.config.rope_base_global
            )
        else:
            pos_embed_sparse = pos_embed_global = _fixed_pos_embedding(
                batch["ages"], head_dim, x.dtype, base=self.config.rope_base_global
            )

        base_mask = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            batch["patient_lengths"].tolist()
        )
        local_bias = base_mask.make_local_attention(self.config.attention_width)
        dense_bias = base_mask.make_causal()

        normed_ages = batch["normalized_ages"]
        for i, layer in enumerate(self.layers):
            is_dense = self._dense_mask[i]
            bias = dense_bias if is_dense else local_bias
            pos_embed = pos_embed_global if is_dense else pos_embed_sparse
            x = x + layer(x, normed_ages, pos_embed, bias)

        return self.out_norm(x)


# -----------------------------------------------------------------------------
# Top‑level Model (transformer backbone + Task head)
# -----------------------------------------------------------------------------
class EHRFM(transformers.PreTrainedModel):
    config_class = EHRFMConfig
    main_input_name = "input_ids"  # For FLOP Counting

    def __init__(self, config: EHRFMConfig, **kwargs):
        super().__init__(config)
        # Allow overriding task config at runtime (Trainer needs this)
        if "task" in kwargs:
            config.task = kwargs["task"]

        self.main_input_name = "input_ids"

        self.transformer = DenseTransformer(config.transformer)

        if config.task is not None:
            self.task_head = make_task_head(hidden_size=config.transformer.hidden_size, **config.task)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_loss=True,
        return_logits=False,
        return_reprs=False,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        features = self.transformer(batch)
        features = features.view(-1, features.shape[-1])
        label_idx = batch["label_indices"]
        task_features = features[label_idx]
        loss, result = self.task_head(task_features, batch["task"], return_logits=return_logits)
        if return_reprs:
            result["representations"] = task_features
        if return_loss:
            return loss, result
        else:
            return result
