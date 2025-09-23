
from pydantic import BaseModel


class Config(BaseModel):
    max_tokens: int = 512
    device: str = 'cuda'
    cache_dir: str = "./.models"
    model_tag: str = "colbert-ir/colbertv2.0"
    max_batch_size: int = 798
    top_k: int = 5