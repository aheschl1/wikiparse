import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase
from abc import ABC, abstractmethod

from wikindex.config import Config

class Encoder:
    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def tokenize(self, texts: list[str]) -> dict:
        pass
    
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        pass

class ColBertV2(Encoder):
    def __init__(
        self, 
        config: Config = Config(),
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_tag, cache_dir=config.cache_dir)
        self.model = AutoModel.from_pretrained(config.model_tag, cache_dir=config.cache_dir).to(config.device)
        self.device = config.device
        self.max_tokens = config.max_tokens
        self.config = config
        
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer
        
    def tokenize(self, texts: list[str]) -> dict:
        return self._tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        )

    def encode(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            raise ValueError("No texts to encode.")
        output = None
        for i in range(0, len(texts), self.config.max_batch_size):
            inputs = self.tokenize(texts[i:i+self.config.max_batch_size])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            if output is None:
                output = outputs.last_hidden_state.cpu().detach()
            else:
                output = torch.cat((output, outputs.last_hidden_state.cpu().detach()), dim=0)
        return output # type: ignore     is logically sound

class Scorer(ABC):
    
    def __init__(self, config: Config = Config()):
        self.config = config
    
    @abstractmethod
    def score(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        pass
    
class ColBertScorer(Scorer):
    def score(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Q -> R^BxNxD
        V -> R^BxMxD
        QV^T -> R^BxNxM
        max -> R^BxM
        sum -> R^B
        
        After mul, index i, j is query token i similarity with doc token j
        Then, take the max of each query token ---- which document token is it most similar to?
        The, sum across all query tokens ---- how similar are the query tokens to the document overall?
        """
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(0)
        if doc_embeddings.dim() == 2:
            doc_embeddings = doc_embeddings.unsqueeze(0)

        scores = torch.matmul(query_embedding, doc_embeddings.permute(0, 2, 1))
        return scores.max(dim=-1).values.sum(dim=-1)