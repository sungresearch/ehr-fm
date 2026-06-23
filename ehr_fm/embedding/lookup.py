import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from torch import nn

from ehr_fm.logger import setup_logging

logger = setup_logging(child_name="embedding.lookup")


class EmbeddingLookup:
    """Loads a pre-built embedding lookup for use during training.

    Artifacts are expected in ``artifact_dir``:
      - ``embedding_table.safetensors`` -- (N, D) float16 tensor
      - ``id_mapping.json`` -- embedding_text -> int mapping
      - ``metadata.json`` -- build metadata
    """

    def __init__(self, artifact_dir: Path | str):
        artifact_dir = Path(artifact_dir)

        with open(artifact_dir / "id_mapping.json") as f:
            self.text_to_id: dict[str, int] = json.load(f)

        tensors = load_file(str(artifact_dir / "embedding_table.safetensors"))
        self.embedding_table: np.ndarray = tensors["embedding_table"]

        with open(artifact_dir / "metadata.json") as f:
            self.metadata: dict = json.load(f)

        logger.info(
            f"Loaded embedding lookup: "
            f"{self.num_embeddings} embeddings x {self.embedding_dim}d "
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

    def as_torch_embedding(self, freeze: bool = True) -> nn.Embedding:
        """Create an nn.Embedding initialized from the lookup table.

        The returned module uses float32 weights (upcast from the stored
        float16) so it integrates with standard mixed-precision training.
        """
        weights = torch.from_numpy(self.embedding_table.astype(np.float32))
        return nn.Embedding.from_pretrained(weights, freeze=freeze)
