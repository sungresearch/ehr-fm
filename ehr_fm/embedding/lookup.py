import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file
from torch import nn

from ehr_fm.logger import setup_logging

logger = setup_logging(child_name="embedding.lookup")


class TiedEmbedding(nn.Module):
    """An embedding whose strings share rows: string ``i`` reads row ``row_index[i]`` of ``weight``.

    Strings that share a row get one summed gradient and stay equal. The module wraps the given tensors and
    draws no random number, so the parameters built after it keep the initial values that the seed gives them.
    """

    def __init__(self, weight: torch.Tensor, row_index: torch.Tensor, freeze: bool = True):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=not freeze)
        self.register_buffer("row_index", row_index)
        self.embedding_dim = weight.shape[1]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(self.row_index[ids], self.weight)


class EmbeddingLookup:
    """Loads a pre-built embedding lookup for use during training.

    Artifacts are expected in ``artifact_dir``:
      - ``embedding_table.safetensors`` -- (N, D) float16 tensor
      - ``id_mapping.json`` -- embedding_text -> int mapping
      - ``metadata.json`` -- build metadata
      - ``row_index.safetensors`` -- optional (N,) int64 tensor. With it, the table holds K shared rows,
        string ``i`` reads row ``row_index[i]``, and ``embedding_table`` is the (N, D) table read through it.
    """

    def __init__(self, artifact_dir: Path | str):
        artifact_dir = Path(artifact_dir)

        with open(artifact_dir / "id_mapping.json") as f:
            self.text_to_id: dict[str, int] = json.load(f)

        tensors = load_file(str(artifact_dir / "embedding_table.safetensors"))
        self.embedding_table: np.ndarray = tensors["embedding_table"]
        self.row_index: np.ndarray | None = None
        self.tied_table: np.ndarray | None = None
        index_path = artifact_dir / "row_index.safetensors"
        if index_path.exists():
            self.row_index = load_file(str(index_path))["row_index"]
            if self.row_index.shape != (len(self.text_to_id),):
                raise ValueError(f"{index_path}: {len(self.row_index)} rows, {len(self.text_to_id)} ids")
            self.tied_table = self.embedding_table
            self.embedding_table = self.tied_table[self.row_index]

        with open(artifact_dir / "metadata.json") as f:
            self.metadata: dict = json.load(f)

        tied = "" if self.tied_table is None else f", tied to {self.tied_table.shape[0]} rows"
        logger.info(
            f"Loaded embedding lookup: "
            f"{self.num_embeddings} embeddings x {self.embedding_dim}d{tied} "
            f"(model={self.metadata.get('model_name')})"
        )

    @property
    def num_embeddings(self) -> int:
        return self.embedding_table.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.embedding_table.shape[1]

    def get_id(self, text: str) -> int:
        """Map an embedding_text string to its integer ID."""
        return self.text_to_id[text]

    def as_torch_embedding(self, freeze: bool = True) -> nn.Embedding | TiedEmbedding:
        """Create an nn.Embedding from the lookup table, or a TiedEmbedding when the lookup has a row index.

        The returned module uses float32 weights (upcast from the stored
        float16) so it integrates with standard mixed-precision training.
        """
        if self.tied_table is not None:
            weights = torch.from_numpy(self.tied_table.astype(np.float32))
            return TiedEmbedding(weights, torch.from_numpy(self.row_index.astype(np.int64)), freeze=freeze)
        weights = torch.from_numpy(self.embedding_table.astype(np.float32))
        return nn.Embedding.from_pretrained(weights, freeze=freeze)
