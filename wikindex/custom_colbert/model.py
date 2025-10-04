from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase
from abc import ABC, abstractmethod
import torch.nn as nn
import string
from wikindex.config import Config

class Encoder:
    @abstractmethod
    def encode(self, texts: list[str], is_query: bool) -> list[torch.Tensor]:
        pass
    
    @abstractmethod
    def encode_tokens(self, inputs: dict[str, torch.Tensor], is_query: bool) -> list[torch.Tensor]:
            pass
    
    @abstractmethod
    def tokenize(self, texts: list[str], is_query: bool) -> dict:
        pass
    
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        pass

class ColBert(Encoder):
    def __init__(
        self, 
        tokenizer_only: bool = False,
        config: Config = Config(),
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_tag, 
            cache_dir=config.cache_dir,
            use_fast=True
        )
        self._query_prepend = self._tokenizer.convert_tokens_to_ids("[unused0]")
        self._document_prepend = self._tokenizer.convert_tokens_to_ids("[unused1]")
        self._cls_token_id = self._tokenizer.cls_token_id
        self._sep_token_id = self._tokenizer.sep_token_id
        self._mask_token_id = self._tokenizer.mask_token_id
        if not tokenizer_only:
            self.model = AutoModel.from_pretrained(config.model_tag, cache_dir=config.cache_dir).to(config.device)
            # the projection is not included in the pretrained model, so we have to load it separately
            self._projection = nn.Linear(768, 128, bias=False, device=config.device)
            state_dict = torch.load(
                hf_hub_download(config.model_tag, "pytorch_model.bin"), 
                map_location=config.device
            )
            self._projection.weight.data.copy_(state_dict["linear.weight"])


        else:
            self.model = None
        self.device = config.device
        self.max_tokens = config.max_tokens
        self.config = config
        vocab = self.tokenizer.get_vocab()
        self._skip_list = torch.tensor([vocab[p] for p in string.punctuation if p in vocab] + [])

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer
        
    def tokenize(self, texts: list[str], is_query: bool) -> dict:
        tokens = [self._tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_tokens-3,
            add_special_tokens=False,
        ) for text in texts]
        
        prefix = [self._cls_token_id, self._query_prepend] if is_query else [self._cls_token_id, self._document_prepend]
        suffix = [self._sep_token_id]

        prefix, suffix = [self._cls_token_id, self._query_prepend], [self._sep_token_id]
        tokens = torch.tensor([prefix + lst["input_ids"] + suffix + [self._mask_token_id] * (self.config.max_tokens - (len(lst["input_ids"])+3)) for lst in tokens])
        attention_mask = (tokens != self._tokenizer.pad_token_id).long()
        return {"input_ids": tokens, "attention_mask": attention_mask}

    def encode_tokens(self, inputs: dict[str, torch.Tensor], is_query: bool) -> list[torch.Tensor]:
        """
        Encode pre-tokenized inputs
        
        Returns a tensor of shape (B, max_tokens, hidden_size)
        """
        assert self.model is not None, "Model not loaded."
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs: torch.Tensor = self.model(**inputs).last_hidden_state
            outputs = self._projection(outputs)
            # l2 normalize the output embeddings
            # shape [B, N, D]
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1).cpu().detach()
            if not is_query:
                punctuation_mask = ~torch.isin(inputs['input_ids'].cpu(), self._skip_list)
                output_masked = [d[punctuation_mask[i]] for i, d in enumerate(outputs)]
            else:
                output_masked = [d for d in outputs]
        return output_masked
    
    def encode(self, texts: list[str], is_query: bool) -> list[torch.Tensor]:
        """
        Encode texts in batches
        
        Returns a tensor of shape (len(texts), max_tokens, hidden_size)
        """
        assert self.model is not None, "Model not loaded."
        if len(texts) == 0:
            raise ValueError("No texts to encode.")
        output = []
        for i in range(0, len(texts), self.config.max_batch_size):
            inputs = self.tokenize(texts[i:i+self.config.max_batch_size], is_query=is_query)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs: torch.Tensor = self.model(**inputs).last_hidden_state
                outputs = self._projection(outputs)
                # l2 normalize the output embeddings
                # shape [B, N, D]
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1).cpu().detach()
                if not is_query:
                    punctuation_mask = ~torch.isin(inputs['input_ids'].cpu(), self._skip_list)
                    output_masked = [d[punctuation_mask[i]] for i, d in enumerate(outputs)]
                else:
                    output_masked = [d for d in outputs]
                    
            output.extend(output_masked)
        return output # type: ignore     is logically sound
    
class Scorer(ABC):
    
    def __init__(self, config: Config = Config()):
        self.config = config
    
    @abstractmethod
    def score(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pad_embeddings(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        pass

class ColBertScorer(Scorer):
    
    def pad_embeddings(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(e.shape[0] for e in embeddings)
        padded = torch.zeros((len(embeddings), max_len, embeddings[0].shape[1]), dtype=embeddings[0].dtype, device=embeddings[0].device)
        for i, e in enumerate(embeddings):
            padded[i, :e.shape[0], :] = e
        return padded
    
    def score(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor: # type: ignore
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
        scores = torch.matmul(query_embedding, doc_embeddings.permute(0, 2, 1))
        return scores.max(dim=-1).values.sum(dim=-1)