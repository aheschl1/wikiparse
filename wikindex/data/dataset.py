
from wikindex.config import Config
from wikindex.custom_colbert.model import Encoder
from wikindex.wiki.parse import WikiSet
from pathlib import Path    


class Dataset:
    def __init__(self, texts: list[str]):
        self.texts = texts
        self._embeddings = None
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
    def embed(self, encoder: Encoder):
        self._embeddings = encoder.encode(self.texts)

    @property
    def embeddings(self):
        if self._embeddings is None:
            raise ValueError("Embeddings have not been computed. Call embed() first.")
        return self._embeddings
    
class WikiDataset(Dataset):
    def __init__(self, config: Config, root: Path, encoder: Encoder):
        self.config = config
        self.wiki_set = WikiSet(root, encoder = encoder, config=config)
        texts = [chunk for _, _, chunk in self.wiki_set.chunks()]
        super().__init__(texts)