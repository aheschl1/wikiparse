
from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Union

from pydantic import BaseModel
from wikindex.config import Config
from wikindex.custom_colbert.model import Encoder
from pathlib import Path

class Datapoint(ABC, BaseModel):
    
    text: str
    id: Union[int, None]
    
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

    def __getitem__(self, idx) -> Union[Datapoint, list[Datapoint]]:
        return self.datapoints[idx]
    
    def __iter__(self):
        yield from self.datapoints.values()

    def embed(self, encoder: Encoder):
        return encoder.encode(self.texts, is_query=False)

    @property
    def texts(self) -> list[str]:
        return [dp.text for dp in self]
    
