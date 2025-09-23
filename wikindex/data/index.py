

from wikindex.config import Config
from wikindex.data.dataset import Dataset
from wikindex.custom_colbert.model import Encoder, Scorer


class Index:
    def __init__(self, dataset: Dataset, encoder: Encoder, scorer: Scorer, config: Config = Config()):
        self.encoder = encoder
        self.dataset = dataset
        self.scorer = scorer
        self.config = config
        self.dataset.embed(self.encoder)

    def search(self, query: str, top_k=None):
        if not top_k:
            top_k = self.config.top_k
        query_embedding = self.encoder.encode([query])
        scores = self.scorer.score(query_embedding, self.dataset.embeddings)
        scores, indices = scores.topk(top_k)
        return scores, [self.dataset[i] for i in indices.tolist()]
