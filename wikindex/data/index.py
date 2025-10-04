

import logging
from pathlib import Path
from typing import Type, Union

import click
import torch
from wikindex.config import Config
from wikindex.data.dataset import Datapoint, Dataset
from wikindex.custom_colbert.model import ColBert, ColBertScorer, Encoder, Scorer
import faiss

from wikindex.wiki.sqlite import DEFAULT_DB_URL, Chunk, Client
from wikindex.wiki.wiki import ChunkDatapoint, WikiDataset
import numpy as np
from tqdm import tqdm
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
        query_embedding = self.encoder.encode([query], is_query=True)
        scores = self.scorer.score(self.scorer.pad_embeddings(query_embedding), self.scorer.pad_embeddings(self.embeddings))
        scores, indices = scores.topk(top_k)
        return scores, [self.dataset[i] for i in indices.tolist()]
    
class FaissIndex(Index):
    def __init__(
        self, 
        dataset: Dataset, 
        encoder: Encoder, 
        scorer: Scorer,
        shard_manager: "ShardManager",
        embeddings_path: Path,
        client: Client | None = None,
        config: Config = Config()
    ):
        super().__init__(dataset, encoder, scorer, config)
        self.shard_manager = shard_manager
        self.embeddings_folder = embeddings_path
        self.client = client
        self._index = None

    def _shard_batch(
        self, 
        datapoints: list[Datapoint]
    ):
        # tensors = [torch.load(self.embeddings_folder / f"{dp.id}.pt") for dp in datapoints]
        tensors = []
        for dp in datapoints:
            try:
                tensor = torch.load(self.embeddings_folder / f"{dp.id}.pt")
                tensors.append(tensor)
            except Exception as e:
                logging.warning(f"Could not load embedding for datapoint {dp.id}: {e}")
                
        catted = torch.cat(tensors, dim=0).cpu()
        dp_ids = []
        for i, dp in enumerate(datapoints):
            dp_ids.extend([dp.id] * tensors[i].shape[0])
        indices = torch.tensor(dp_ids, device=catted.device)
        shard_ids = self.shard_manager.assign_shards(catted, nprobe=4)
        shard_ids = torch.tensor(shard_ids, device=catted.device)
        for sid in torch.unique(shard_ids):
            to_add = catted[shard_ids == sid, :]
            dids = indices[shard_ids == sid]
            index = self.shard_manager.load_shard(int(sid))
            index.add_with_ids(to_add.numpy(), dids.numpy()) # type: ignore
            self.shard_manager.close_shard(index, int(sid))
        # ids = []
        # for i, dp in enumerate(datapoints):
        #     ids.extend([dp.id] * tensors[i].shape[0])
        # indices = torch.tensor(ids, device=catted.device)
        # index.add_with_ids(catted.numpy(), indices.numpy()) # type: ignore
    
    def build(self, train: bool = True):
        if self.client is None:
            raise ValueError("Client must be provided to shard the index.")
        if train:
            self.shard_manager.train(self.dataset, self.embeddings_folder)
            logging.info("Index trained.")        

        length = len(self.dataset)
        batch_size = self.config.max_batch_size
        logging.info(f"Building index for {length} datapoints.")
        batch = []
        for datapoint in tqdm(self.dataset, total=length, desc="Building Faiss index"):
            batch.append(datapoint)
            if len(batch) >= batch_size:
                self._shard_batch(batch)
                batch = []
                
        if batch:
            self._shard_batch(batch)
        logging.info("Sharding complete.")

    @staticmethod
    def load(
        dataset: Dataset,
        encoder: Encoder,
        scorer: Scorer,
        embeddings_path: Path,
        root: Path,
        config: Config = Config()
    ) -> "FaissIndex":
        manager = ShardManager(root)
        manager.load()
        findex = FaissIndex(
            dataset=dataset,
            encoder=encoder,
            scorer=scorer,
            embeddings_path=embeddings_path,
            shard_manager=manager,
            config=config
        )
        return findex

    def search(self, query: str, top_k=None):
        if not top_k:
            top_k = self.config.top_k
        embeddings = self.encoder.encode([query], is_query=True)[0].cpu()
        shard_ids = self.shard_manager.assign_shards(embeddings, nprobe=1)
        all_indices = set()
        for sid in np.unique(shard_ids):
            index = self.shard_manager.load_shard(int(sid))
            _, indices = index.search(embeddings.numpy()[shard_ids == sid], self.config.top_k_prefetch) # type: ignore
            all_indices.update(indices[indices != -1].flatten().tolist())
            self.shard_manager.close_shard(index, int(sid))

        chunks = []
        all_indices = list(all_indices)
        for did in all_indices:
            chunk = torch.load(self.embeddings_folder / f"{did}.pt", map_location=self.config.device)
            chunks.append(chunk)
        scores = self.scorer.score(embeddings.to(torch.float16).to(self.config.device), self.scorer.pad_embeddings(chunks))
        scores, indices = scores.topk(top_k)
        return scores, [self.dataset[all_indices[i]] for i in indices.tolist()]

class ShardManager:
    def __init__(
        self, 
        root: Path, 
        config: Config = Config()
    ):
        self.config = config
        self.root = root
        self.quantizer_index_path = root / "quantizer.index"
        self._quantizer: faiss.IndexIVFPQ | None = None
        self.nlist = 4096        # number of coarse clusters (tunable)
        self.m = 32              # subquantizers (dim should be divisible by m)
        self.nbits = 8           # bits per subquantizer (2^8 = 256 centroids each)
        self.ntrain = 100_000    

    @property
    def quantizer(self):
        if self._quantizer is None:
            raise ValueError("Quantizer not loaded. Call load() or train() first.")
        return self._quantizer.quantizer
    
    def train(self, dataset: Dataset, embeddings_folder: Path):
        
        
        quantizer = faiss.IndexFlatIP(self.config.projected_dim)
        index = faiss.IndexIVFPQ(quantizer, self.config.projected_dim, self.nlist, self.m, self.nbits)

        # Train IVF-PQ on a sample of doc embeddings
        logging.info("Sampling embeddings for PQ training...")
        sample_indices = np.random.choice(len(dataset), size=min(self.ntrain, len(dataset)), replace=False)
        train_samples = []
        failures = 0
        print(f"Success: 0 | Failures: 0 | Remaining: {len(sample_indices)} | Total: {len(sample_indices)}", end="\r")
        for idx in sample_indices:
            idx = max(1, int(idx)) # the assigned sql ids start at 1
            try:
                train_samples.append(torch.load(embeddings_folder / f"{idx}.pt"))
            except Exception as e:
                failures += 1
            # remove old print statement
            print(f"Success: {len(train_samples)} | Failures: {failures} | Remaining: {len(sample_indices) - len(train_samples) - failures} | Total: {len(sample_indices)}", end="\r")

        logging.info(f"Loaded {len(train_samples)} training samples, {failures} failures.")
        
        sample = torch.cat(train_samples, dim=0).numpy()
        logging.info(f"Training index on {sample.shape[0]} samples...")
        index.train(sample)  # type: ignore
        
        faiss.write_index(index, str(self.quantizer_index_path)) # type: ignore
        logging.info(f"Saved quantizer to {self.quantizer_index_path}.")
        self._quantizer = index
        
    def load(self):
        self._quantizer = faiss.read_index(str(self.quantizer_index_path))
        
    def assign_shards(self, embeddings: torch.Tensor, nprobe=1) -> np.ndarray:
        if self._quantizer is None:
            raise ValueError("Quantizer not loaded. Call load() or train() first.")
        I = self._quantizer.quantizer.assign(embeddings.cpu().numpy(), k=nprobe) # type: ignore
        return I.reshape(-1)
    
    def load_shard(self, shard_id: int) -> faiss.Index:
        shard_path = self.root / f"shard_{shard_id}.index"
        if not shard_path.exists():
            index = faiss.clone_index(self._quantizer) # type: ignore
            index.reset()
            index = faiss.IndexIDMap(index)
            faiss.write_index(index, str(shard_path))
        return faiss.read_index(str(shard_path))
    
    def close_shard(self, index: faiss.Index, shard_id: int):
        shard_path = self.root / f"shard_{shard_id}.index"
        faiss.write_index(index, str(shard_path))
        del index
        


@click.command()
@click.argument("root", type=click.Path(file_okay=False, path_type=Path))
@click.argument("embeddings_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db-url", type=str, default=DEFAULT_DB_URL, help="Database URL for SQLite.")
@click.option("--train", is_flag=True, default=False, help="Whether to train the quantizer.")
def main(root: Path, embeddings_path: Path, db_url: str, train: bool):
    """
    Uses chunk table to build an index and perform a sample search.
    """
    if not root.exists():
        if not train:
            raise ValueError(f"Root path {root} does not exist. Cannot load index. Enable --train to create a new index.")
        root.mkdir(parents=True)
    if not db_url.startswith("sqlite:///"):
        db_url = "sqlite:///" + db_url
    logging.basicConfig(level=logging.INFO)
    config = Config(device="cuda")
    encoder = ColBert(config=config, tokenizer_only=False)
    scorer = ColBertScorer(config=config)
    with Client(db_url) as client:
        dataset = WikiDataset.from_sqlite(client, encoder, config)
        shard_manager = ShardManager(root)
        if not train:
            shard_manager.load()
        index = FaissIndex(
            dataset, 
            encoder,
            scorer, 
            embeddings_path=embeddings_path,
            config=config,
            shard_manager=shard_manager,
            client=client
        )
        index.build(train=train)
        # index.shard_index()
        # index = FaissIndex.load(dataset, encoder, scorer, config=config, index_path="./faiss.index")
        # while True:
        #     q = input("Enter query (or 'exit' to quit): ")
        #     if q.lower() in ("exit", "quit"):
        #         break
        #     print(index.search(q))

if __name__ == "__main__":
    main()
    # ShardManager(Path("faiss.quant"))