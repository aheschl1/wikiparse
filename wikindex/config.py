
from pydantic import BaseModel


class Config(BaseModel):
    max_tokens: int = 512
    device: str = 'cuda'
    cache_dir: str = "./.models"
    model_tag: str = "colbert-ir/colbertv2.0"
    max_batch_size: int = 950
    top_k: int = 10
    top_k_prefetch: int = 1024
    projected_dim: int = 128