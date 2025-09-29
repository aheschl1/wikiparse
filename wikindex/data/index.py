

import logging
from pathlib import Path
from typing import Type, Union

import click
import torch
from wikindex.config import Config
from wikindex.data.dataset import Datapoint, Dataset
from wikindex.custom_colbert.model import ColBert, ColBertScorer, Encoder, Scorer
import faiss

from wikindex.wiki.sqlite import DEFAULT_DB_URL, Chunk, Client, Shard
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

    def _classify_batch(
        self, 
        datapoints: list[Datapoint]
    ):
        tensors = [torch.load(self.embeddings_folder / f"{dp.id}.pt") for dp in datapoints]
        catted = torch.cat(tensors, dim=0).cpu()
        shard_ids = self.shard_manager.assign_shards(catted, nprobe=4)
        return shard_ids
        # ids = []
        # for i, dp in enumerate(datapoints):
        #     ids.extend([dp.id] * tensors[i].shape[0])
        # indices = torch.tensor(ids, device=catted.device)
        # index.add_with_ids(catted.numpy(), indices.numpy()) # type: ignore
    
    def shard_index(self):
        if self.client is None:
            raise ValueError("Client must be provided to shard the index.")
        self.shard_manager.train(self.dataset, self.embeddings_folder)
        logging.info("Index trained.")        

        length = len(self.dataset)
        batch_size = self.config.max_batch_size
        logging.info(f"Building index for {length} datapoints.")
        batch = []
        for datapoint in tqdm(self.dataset, total=length, desc="Building Faiss index"):
            batch.append(datapoint)
            if len(batch) >= batch_size:
                shard_ids = self._classify_batch(batch)
                self.client.session.add_all([
                    Shard(chunk_id=datapoint.id, shard_id=int(shard_id)) for shard_id, datapoint in zip(shard_ids, batch)
                ])
                self.client.session.commit()
                batch = []
                
        if batch:
            shard_ids = self._classify_batch(batch)
            self.client.session.add_all([
                Shard(chunk_id=datapoint.id, shard_id=int(shard_id)) for shard_id, datapoint in zip(shard_ids, batch)
            ])
            self.client.session.commit()
        logging.info("Sharding complete.")
    
    def _index_batch(
        self, 
        datapoints: list[Datapoint],
        index: faiss.Index,
    ):
        tensors = [torch.load(self.embeddings_folder / f"{dp.id}.pt") for dp in datapoints]
        catted = torch.cat(tensors, dim=0).cpu()
        ids = []
        for i, dp in enumerate(datapoints):
            ids.extend([dp.id] * tensors[i].shape[0])
        indices = torch.tensor(ids, device=catted.device)
        index.add_with_ids(catted.numpy(), indices.numpy()) # type: ignore
    
    def _index_shard(self, shard_id: int):
        assert self.client is not None, "Client must be provided to load the index."
        index = self.shard_manager.load_shard(shard_id)
        if index is None:
            logging.warning(f"Shard {shard_id} not found.")
            return
        # Load the corresponding datapoints for this shard
        query = self.client.session.query(Chunk).join(Shard, Chunk.id == Shard.chunk_id).filter(Shard.shard_id == shard_id)
        batch = []
        for row in query.yield_per(100):
            datapoint = ChunkDatapoint(text=row.content, id=row.id, doc_id=row.doc_id) # type: ignore
            batch.append(datapoint)
            if len(batch) >= self.config.max_batch_size:
                self._index_batch(batch, index)
                batch = []
        if batch:
            self._index_batch(batch, index)

        self.shard_manager.close_shard(index, shard_id)

    def build_index(self):
        if self.client is None:
            raise ValueError("Client must be provided to shard the index.")
        unique_shards = self.client.session.query(Shard.shard_id).distinct().all() 
        shard_ids = [shard_id for (shard_id,) in unique_shards]
        logging.info(f"Loading {len(shard_ids)} shards into Faiss index.")
        for i, shard_id in enumerate(tqdm(shard_ids, desc="Loading shards")):
            # for each shard id, we need to populate it
            self._index_shard(shard_id)

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
        assert self._index is not None, "Index has not been built or loaded."
        if not top_k:
            top_k = self.config.top_k

        prefetch_n = self.config.top_k_prefetch
        query_embedding = self.encoder.encode([query], is_query=True)[0].cpu().numpy()
        _, I = self._index.search(query_embedding, prefetch_n) # type: ignore
        ids = np.unique(I)
        embeddings = [torch.load(self.embeddings_folder / f"{id}.pt") for id in ids]
        d = self.scorer.pad_embeddings(embeddings).to(self.config.device)
        q = torch.from_numpy(query_embedding).to(self.config.device).to(d.dtype)
        scores = self.scorer.score(q, d)
        scores, i = scores.topk(top_k)
        return scores, [self.dataset[id] for id in ids[i.cpu().tolist()].tolist()]

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

    @property
    def quantizer(self):
        if self._quantizer is None:
            raise ValueError("Quantizer not loaded. Call load() or train() first.")
        return self._quantizer.quantizer
    
    def train(self, dataset: Dataset, embeddings_folder: Path):
        nlist = 4096        # number of coarse clusters (tunable)
        m = 32              # subquantizers (dim should be divisible by m)
        nbits = 8           # bits per subquantizer (2^8 = 256 centroids each)
        ntrain = 70_000    
        
        quantizer = faiss.IndexFlatIP(self.config.projected_dim)
        index = faiss.IndexIVFPQ(quantizer, self.config.projected_dim, nlist, m, nbits)
        
        # Train IVF-PQ on a sample of doc embeddings
        logging.info("Sampling embeddings for PQ training...")
        sample_indices = np.random.choice(len(dataset), size=min(ntrain, len(dataset)), replace=False)
        train_samples = []
        for idx in sample_indices:
            idx = max(1, int(idx)) # the assigned sql ids start at 1
            train_samples.append(torch.load(embeddings_folder / f"{idx}.pt"))

        sample = torch.cat(train_samples, dim=0).numpy()
        logging.info(f"Training index on {sample.shape[0]} samples...")
        index.train(sample)  # type: ignore
        
        faiss.write_index(index, str(self.quantizer_index_path)) # type: ignore
        logging.info(f"Saved quantizer to {self.quantizer_index_path}.")
        self._quantizer = index
        
    def load(self):
        self._quantizer = faiss.read_index(str(self.quantizer_index_path))
        
    def assign_shards(self, embeddings: torch.Tensor, nprobe=4) -> np.ndarray:
        if self._quantizer is None:
            raise ValueError("Quantizer not loaded. Call load() or train() first.")
        I = self._quantizer.quantizer.assign(embeddings.cpu().numpy(), k=1) # type: ignore
        return I.reshape(-1)
    
    def load_shard(self, shard_id: int) -> faiss.Index:
        shard_path = self.root / f"shard_{shard_id}.index"
        if not shard_path.exists():
            raise ValueError(f"Shard {shard_id} does not exist at {shard_path}.")
        return faiss.read_index(str(shard_path))
    
    def close_shard(self, index: faiss.Index, shard_id: int):
        shard_path = self.root / f"shard_{shard_id}.index"
        faiss.write_index(index, str(shard_path))
        logging.info(f"Saved shard {shard_id} to {shard_path}.")
        index.reset()
        


@click.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("embeddings_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db-url", type=str, default=DEFAULT_DB_URL, help="Database URL for SQLite.")
def main(root: Path, embeddings_path: Path, db_url: str):
    """
    Uses chunk table to build an index and perform a sample search.
    """
    if not db_url.startswith("sqlite:///"):
        db_url = "sqlite:///" + db_url
    logging.basicConfig(level=logging.INFO)
    config = Config(device="cuda")
    encoder = ColBert(config=config, tokenizer_only=False)
    scorer = ColBertScorer(config=config)
    with Client(db_url) as client:
        dataset = WikiDataset.from_sqlite(client, encoder, config)
        index = FaissIndex(
            dataset, 
            encoder,
            scorer, 
            embeddings_path=embeddings_path,
            config=config,
            shard_manager=ShardManager(root),
            client=client
        )
        index.shard_index()
        # index = FaissIndex.load(dataset, encoder, scorer, config=config, index_path="./faiss.index")
        # print(index.search("what is a frog?"))

if __name__ == "__main__":
    main()
    # ShardManager(Path("faiss.quant"))