from pathlib import Path
from re import compile
import re
from threading import Thread
import time
from typing import Literal, Union
from urllib.parse import unquote
from pydantic import BaseModel, Field
import html
from glob import glob
import click
import sqlalchemy
import torch

from wikindex.wiki.sqlite import DEFAULT_DB_URL, TRANSACTION_BATCH_SIZE, Client, Document, Chunk, FileRecord, document_links

from wikindex.config import Config
from wikindex.custom_colbert.model import ColBert, Encoder

import logging

from wikindex.data.dataset import Datapoint, Dataset

from multiprocessing import Queue, Process, Manager
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Task(BaseModel):
    task: Literal["doc", "chunk", "file"]
    content: Union["WikiDoc", "ChunkDatapoint", str]
    
class ChunkDatapoint(Datapoint, BaseModel):
    text: str
    id: Union[int, None]
    doc_id: Union[int, None] = None

    @property
    def metadata(self) -> dict:
        return {}

class WikiDoc(BaseModel):
    id: int = Field(..., description="The unique identifier for the document.")
    url: str = Field(..., description="The URL of the document.")
    title: str = Field(..., description="The title of the document.")
    chunks: list[str] = Field(..., description="The main content of the document, split into chunks.")
    links: list[str] = Field(default_factory=list, description="List of titles of referenced docs.")
    file_path: str = Field(default="", description="The file path where the document is located.")
    
    def get_datapoints(self):
        for chunk in self.chunks:
            yield ChunkDatapoint(text=chunk, id=None, doc_id=self.id)

    def __len__(self):
        return len(self.chunks)

    @staticmethod
    def chunk_content(
        clean_content: str,
        title: str,
        encoder: "Encoder",
        config: "Config" = Config()
    ) -> list[str]:
        tokenizer = encoder.tokenizer
        max_tokens = config.max_tokens

        # Stage 1: split by newline (skip title)
        paragraphs = [
            chunk.strip()
            for chunk in clean_content.split("\n\n")
            if chunk.strip() and chunk.strip() != title
        ]

        chunks: list[str] = []

        # Stage 2: process each chunk
        for chunk in paragraphs:
            tokens = tokenizer.tokenize(chunk)

            if len(tokens) <= max_tokens - 2:
                # already fits, keep it
                chunks.append(chunk)
                continue

            # split into sentences and repack greedily
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            buffer: list[str] = []

            for sentence in sentences:
                sent_tokens = tokenizer.tokenize(sentence)

                if len(buffer) + len(sent_tokens) <= max_tokens - 2:
                    buffer.extend(sent_tokens)
                else:
                    if buffer:
                        # re-encode so we can get back to text
                        chunks.append(tokenizer.convert_tokens_to_string(buffer))
                    buffer = sent_tokens

            if buffer:
                chunks.append(tokenizer.convert_tokens_to_string(buffer))

        return chunks
    
DOC_PATTERN = re.compile(
    r'<doc id="(?P<id>\d+)" url="(?P<url>[^"]+)" title="(?P<title>[^"]+)">(?P<content>.*?)</doc>',
    re.DOTALL
)
LINK_PATTERN = re.compile(r'<a href="([^"]+)">(.*?)</a>')


def clean_content(raw: str) -> tuple[str, list[str]]:
    """Unescape HTML, extract references, and strip tags."""
    text = html.unescape(raw)
    references: list[str] = []

    def replace_link(match: re.Match) -> str:
        href, link_text = match.groups()
        references.append(href)
        return link_text

    text = LINK_PATTERN.sub(replace_link, text)
    text = re.sub(r'<[^>]+>', '', text)  # remove other HTML tags
    return text, references



class WikiFile:
    def __init__(self, path: Path, encoder: Encoder, tasks_queue: Queue, config: Config = Config()):
        self._path = path
        self.config = config
        self.encoder = encoder
        self._tasks_queue = tasks_queue

    @property
    def path(self) -> str:
        return f"{self._path.parent.name} / {self._path.name}"
    
    def _load_text(self) -> str:
        return self._path.read_text(encoding='utf-8')

    def parse(self):
        title_to_id = {}
        for match in DOC_PATTERN.finditer(self._load_text()):
            doc_id, url, title, raw_content = (
                int(match["id"]),
                match["url"],
                match["title"],
                match["content"].strip(),
            )
            content, references = clean_content(raw_content)
            doc = WikiDoc(
                id=doc_id,
                url=url,
                title=title,
                chunks=WikiDoc.chunk_content(content, title, self.encoder, self.config),
                links=references,
                file_path=self.path
            )
            self._tasks_queue.put(Task(task="doc", content=doc))
            for datapoint in doc.get_datapoints():
                self._tasks_queue.put(Task(task="chunk", content=datapoint))
            title_to_id[title] = doc_id
        return title_to_id

class WikiDataset(Dataset):
    def __init__(
        self, 
        root: Path, 
        tasks_queue: Union[Queue, None],
        client: Client,
        encoder: Encoder, 
        consumer_processes: int = 1,
        config: Config = Config()
    ):
        self.root = root
        self.config = config
        self.encoder = encoder
        self._client = client
        self._tasks_queue = tasks_queue
        self.consumer_processes = consumer_processes
        self._datapoints = None
    
    @staticmethod
    def from_sqlite(
        client: Client,
        encoder: Encoder,
        config: Config = Config()
    ) -> "WikiDataset":
        return WikiDataset(
            root=Path("."),
            tasks_queue=None,
            client=client,
            encoder=encoder,
            config=config
    )
    
    def parse(self):
        assert self._tasks_queue is not None, "Tasks queue must be provided for parsing."
        def process_file(path: Path):
            wf = WikiFile(
                path, encoder=self.encoder,
                tasks_queue=self._tasks_queue, # type: ignore
                config=self.config
            )
            wf.parse()
            self._tasks_queue.put(Task( # type: ignore
                task="file",
                content=wf.path
            ))

        paths = [Path(x) for x in glob(str(self.root / "*" / "*_*"))]
        total_path = len(paths)
        paths = [p for p in paths if not self._client.session.get(FileRecord, f"{p.parent.name} / {p.name}")]
        logger.info(f"Found {len(paths)} / {total_path} files to process in {self.root}")
        
        with ThreadPool(processes=self.consumer_processes) as pool, tqdm(total=len(paths), desc="Processing files") as pbar:
            for _ in pool.imap(process_file, paths):
                pbar.update()
                pbar.refresh()
    
    def __getitem__(self, idx) -> Union[Datapoint, list[Datapoint]]:
        if isinstance(idx, int):
            row: Chunk = self._client.session.get(Chunk, idx)
            if not row:
                raise IndexError(f"Index {idx} out of range.")
            return ChunkDatapoint(text=row.content, id=row.id, doc_id=row.doc_id) # type: ignore
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.__len__()
            stop = min(stop, self.__len__())
            if start < 0 or stop < 0 or start >= self.__len__() or stop > self.__len__() or start >= stop:
                raise IndexError(f"Slice {idx} out of range.")
            rows = self._client.session.execute(
                sqlalchemy.select(Chunk).where(
                    Chunk.id >= start,
                    Chunk.id < stop
                )
            ).scalars().all()
            return [ChunkDatapoint(text=row.content, id=row.id, doc_id=row.doc_id) for row in rows] # type: ignore
        else:
            raise TypeError("Index must be int or slice.")
        
    def __iter__(self):
        for rows in self._client.session.execute(sqlalchemy.select(Chunk)).yield_per(100):
            for row in rows:
                yield ChunkDatapoint(text=row.content, id=row.id, doc_id=row.doc_id)
    
    def __len__(self):
        count = self._client.session.execute(sqlalchemy.select(sqlalchemy.func.count(Chunk.id))).scalar_one()
        return count

    @property
    def datapoints(self) -> dict[int, Datapoint]:
        return {row.id: ChunkDatapoint(text=row.content, id=row.id, doc_id=row.doc_id) for row in self._client.session.execute(sqlalchemy.select(Chunk)).scalars()} # type: ignore

def try_commit_all(session, items):
    try:
        session.add_all(items)
        session.commit()
    except Exception as e:
        logger.error(f"Error adding items: {e}")
        for item in items:
            try:
                session.add(item)
                session.commit()
            except Exception as e:
                logger.error(f"Error adding item {type(item)}: {e}")
                session.rollback()
        session.commit()

def consumer(tasks: Queue, db_url: str):
    with Client(db_url) as client:
        run = True
        while run:
            batch = {
                "doc": [],
                "chunk": [],
                "file": []
            }
            for _ in range(TRANSACTION_BATCH_SIZE):
                task: Task = tasks.get()
                if task is None:
                    run = False
                    break
                batch[task.task].append(task.content)
            if not any(batch.values()):
                break
            if batch["doc"]:
                try_commit_all(client.session, [Document(title=x.title, id=x.id, url=x.url) for x in batch["doc"]])
                # add links
                for doc in batch["doc"]:
                    if not doc.title:
                        raise ValueError("Document title cannot be empty when adding links.")
                    for link in doc.links:
                        if not link:
                            raise ValueError("Link title cannot be empty when adding links.")
                    link_rows = [
                        {"from_title": doc.title, "to_title": unquote(fid)}
                        for fid in doc.links if doc.title and doc.links
                    ]
                    if link_rows:
                        client.session.execute(document_links.insert(), link_rows)
                client.session.commit()
            if batch["chunk"]:
                try_commit_all(client.session, [Chunk(doc_id=x.doc_id, content=x.text) for x in batch["chunk"]])
            if batch["file"]:
                try_commit_all(client.session, [FileRecord(path=fpath) for fpath in batch["file"]])

def preprocess(root: Path, db_url: str = DEFAULT_DB_URL, consumer_processes: int = 1):
    config = Config(device="cpu")
    encoder = ColBert(config=config, tokenizer_only=True)
    
    tasks_queue: Queue = Queue()

    tasks_process = Process(target=consumer, args=(tasks_queue, db_url))
    tasks_process.start()
    
    time.sleep(1)

    logger.info(f"Starting preprocessing of Wiki dataset at {root}")

    with Client(db_url) as client:
        wset = WikiDataset(
            root, 
            encoder=encoder, 
            client=client, 
            tasks_queue=tasks_queue,
            consumer_processes=consumer_processes,
            config=config
        )
        wset.parse()
    # signal consumers to finish
    tasks_queue.put(None)
    tasks_process.join()
    

def _build_batch(datapoints: list[Datapoint], encoder: Encoder, embeddings_folder: Path):
    def write():
        for i, embd in enumerate(embeddings):
            id = datapoints[i].id
            torch.save(embd.cpu().to(torch.float16), embeddings_folder / f"{id}.pt")
    embeddings = encoder.encode([x.text for x in datapoints], is_query=False)
    thread = Thread(target=write)
    thread.start()
    return thread

def embed_dataset(dataset: WikiDataset, encoder: Encoder, output_folder: Path):
    max_threads = 1000
    batch = []
    threads = []
    for datapoint in dataset:
        while len(threads) > max_threads:
            threads.pop(0).join()
            
        batch.append(datapoint)
        if len(batch) >= dataset.config.max_batch_size:
            threads.append(_build_batch(batch, encoder, output_folder))
            batch = []
    if batch:
        threads.append(_build_batch(batch, encoder, output_folder))
    for thread in threads:
        thread.join()

@click.command()
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--db-name', '--db', type=str, default=DEFAULT_DB_URL, help="Database URL or file name for SQLite.")
@click.option('--consumer-processes', '--cp', type=int, default=8, help="Number of consumer processes to use.")
@click.option('--extract', is_flag=True, default=False, help="Whether to extract the sqlite.")
@click.option('--embed', is_flag=True, default=False, help="Whether to embed the dataset after preprocessing.")
def main(root: Path, db_name: str, consumer_processes: int, extract: bool, embed: bool):
    if not db_name.startswith("sqlite:///"):
        db_url = f"sqlite:///{db_name}"
    else:
        db_url = db_name
    if extract:
        preprocess(root, db_url, consumer_processes)
    if embed:
        with Client(db_url) as client:
            config = Config(device="cuda")
            encoder = ColBert(config=config, tokenizer_only=False)
            dataset = WikiDataset.from_sqlite(client, encoder, config)
            embed_dataset(dataset, encoder, root)

        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()