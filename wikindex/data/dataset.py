
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from pydantic import BaseModel
from wikindex.config import Config
from wikindex.custom_colbert.model import Encoder
from pathlib import Path

class Datapoint(ABC, BaseModel):
    
    text: str
    id: int
    
    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass
    
class TextDatapoint(Datapoint):
    def __init__(self, id: int, text: str):
        super().__init__(id=id, text=text)
    @property
    def metadata(self) -> dict:
        return {}

class Dataset:
    def __init__(self, datapoints: Mapping[int, Datapoint]):
        self.datapoints = datapoints
    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> Datapoint:
        return self.datapoints[idx]

    def embed(self, encoder: Encoder):
        return encoder.encode(self.texts)

    @property
    def texts(self) -> list[str]:
        return [dp.text for dp in self.datapoints.values()]
    
