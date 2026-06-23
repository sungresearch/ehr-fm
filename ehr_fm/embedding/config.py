from pathlib import Path

from pydantic import BaseModel


class EmbeddingLookupConfig(BaseModel):
    meds_dir: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    batch_size: int = 64
    device: str = "cuda"
    dtype: str = "float16"
