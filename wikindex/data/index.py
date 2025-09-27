

from typing import Type, Union
from wikindex.config import Config
from wikindex.data.dataset import Dataset
from wikindex.custom_colbert.model import Encoder, Scorer
import faiss

class Index:
    def __init__(self, dataset: Dataset, encoder: Encoder, scorer: Scorer, config: Config = Config()):
        self.encoder = encoder
        self.dataset = dataset
        self.scorer = scorer
        self.config = config
        self.embeddings = None
    
    def build(self):
        self.embeddings = self.dataset.embed(self.encoder)

    def search(self, query: str, top_k=None):
        assert self.embeddings is not None, "Embeddings have not been computed. Call build() first."
        if not top_k:
            top_k = self.config.top_k
        query_embedding = self.encoder.encode([query])
        scores = self.scorer.score(self.scorer.pad_embeddings(query_embedding), self.scorer.pad_embeddings(self.embeddings))
        scores, indices = scores.topk(top_k)
        return scores, [self.dataset[i] for i in indices.tolist()]
    
class FaissIndex(Index):
    def __init__(self, dataset: Dataset, encoder: Encoder, scorer: Scorer, config: Config = Config()):
        super().__init__(dataset, encoder, scorer, config)
        self._qclass = {
            "gpu": faiss.GpuIndexIVFFlat,
            "cpu": faiss.IndexIVFFlat
        }[config.device]
        self._iclass: Union[
            Type[faiss.IndexFlatIP], Type[faiss.GpuIndexFlatIP]
        ] = {
            "gpu": faiss.GpuIndexFlatIP,
            "cpu": faiss.IndexFlatIP
        }[config.device]
        self._index = None

    def build(self):
        assert self.embeddings is not None, "Embeddings have not been computed. Call embed() first."
        quantizer = self._qclass(self.embeddings.shape[-1])
        nlist = 100
        index = self._iclass(quantizer, self.embeddings.shape[-1], nlist, faiss.METRIC_INNER_PRODUCT)

        embeddings = self.embeddings
        ids = self.dataset.datapoints.keys()
        index.train(embeddings.shape[0], embeddings)
        index.add_with_ids(embeddings.shape[0], embeddings, ids)

        self._index = index

    def search(self, query: str, top_k=None):
        assert self._index is not None, "Index has not been built. Call build() first."
        if not top_k:
            top_k = self.config.top_k

        query_embedding = self.encoder.encode([query])
        D, I = self._index.search(
            query_embedding.shape[0], 
            query_embedding, 
            top_k,
            None,
            None
        )
        results = []
        for row_ids, row_scores in zip(I, D):
            results.append([
                (self.dataset.datapoints[i], score)
                for i, score in zip(row_ids, row_scores)
            ])
        
        
        raise NotImplementedError("FaissIndex search not fully implemented.")