from pathlib import Path
from re import compile
import re
from pydantic import BaseModel, Field
import html
from glob import glob
import pandas as pd

from wikindex.wiki.sqlite import DB_URL, Client, Document, Chunk, FileRecord

from wikindex.config import Config
from wikindex.custom_colbert.model import ColBert, Encoder

import logging

from wikindex.data.dataset import Datapoint, Dataset

from multiprocessing import Queue

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ChunkDatapoint(Datapoint, BaseModel):
    text: str
    id: int
    doc_id: int = Field(..., description="The id of the document.")

    @property
    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
        }

class WikiDoc(BaseModel):
    id: int = Field(..., description="The unique identifier for the document.")
    url: str = Field(..., description="The URL of the document.")
    title: str = Field(..., description="The title of the document.")
    chunks: list[str] = Field(..., description="The main content of the document, split into chunks.")
    links: list[str] = Field(default_factory=list, description="List of titles of referenced docs.")
    file_path: str = Field(default="", description="The file path where the document is located.")
    
    def get_datapoints(self, ids: list[int]):
        for i, chunk in enumerate(self.chunks):
            yield ChunkDatapoint(text=chunk, doc_id=self.id, id=ids[i])

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
    def __init__(self, path: Path, encoder: Encoder, config: Config = Config()):
        self._path = path
        self.config = config
        self.encoder = encoder
        self.docs, self.title_to_id = self._parse()

    @property
    def path(self) -> str:
        return f"{self._path.parent.name} / {self._path.name}"
    
    def _load_text(self) -> str:
        return self._path.read_text(encoding='utf-8')

    def _parse(self) -> tuple[dict[int, WikiDoc], dict[int, str]]:
        docs = {}
        title_to_id = {}
        for match in DOC_PATTERN.finditer(self._load_text()):
            doc_id, url, title, raw_content = (
                int(match["id"]),
                match["url"],
                match["title"],
                match["content"].strip(),
            )
            content, references = clean_content(raw_content)
            docs[doc_id] = WikiDoc(
                id=doc_id,
                url=url,
                title=title,
                chunks=WikiDoc.chunk_content(content, title, self.encoder, self.config),
                links=references,
                file_path=self.path
            )
            title_to_id[title] = doc_id
        return docs, title_to_id

class WikiDataset(Dataset):
    def __init__(self, root: Path, encoder: Encoder, config: Config = Config()):
        self.root = root
        self.config = config
        self.encoder = encoder
        self._datapoints = None

        self.files: dict[str, WikiFile] = self._load_files()
        self.docs, self.title_to_id = self._collect_docs()

    def _load_files(self) -> dict[str, WikiFile]:
        files = {}
        for path in map(Path, glob(str(self.root / "*" / "*_*"))):
            wf = WikiFile(
                path, encoder=self.encoder, config=self.config
            )
            files[wf.path] = wf
        return files

    def _collect_docs(self) -> tuple[dict[int, WikiDoc], dict[str, int]]:
        docs = {doc.id: doc for wf in self.files.values() for doc in wf.docs.values()}        
        title_to_id = {}
        for f in self.files.values():
            title_to_id.update(f.title_to_id)
        return docs, title_to_id

    def save(self, output_dir: Path):
        """
        Save mapping of datapoint IDs to (doc title, chunk index) for reference.
        write to csv with columns: id, doc_title, chunk_index
        """
        chunk_data = {
            "id": [],
            "doc_id": []
        }
        doc_data = {
            "id": [],
            "title": [],
            "file_path": []
        }
        for datapoint in self.datapoints.values():
            chunk_data["id"].append(int(datapoint.id))
            chunk_data["doc_id"].append(datapoint.metadata["doc_id"])
        
        for fpath, file in self.files.items():
            for doc in file.docs.values():
                doc_data["id"].append(doc.id)
                doc_data["title"].append(doc.title)
                doc_data["file_path"].append(fpath)
        
        chunk_df = pd.DataFrame(chunk_data)
        doc_df = pd.DataFrame(doc_data)
        chunk_df.to_csv(output_dir / "datapoints.csv", index=False)
        doc_df.to_csv(output_dir / "documents.csv", index=False)
        
    @property
    def datapoints(self) -> dict[int, Datapoint]:
        if self._datapoints:
            return self._datapoints
        points = {}
        base_id = 0
        for doc in self.docs.values():
            ids = list(range(base_id, base_id + len(doc)))
            for dp in doc.get_datapoints(ids):
                points[dp.id] = dp
            base_id += len(doc)
        # cache it
        self._datapoints = points
        return points

BATCH_SIZE = 1000

def document_consumer(docs: Queue[WikiDoc], client: Client):
    run = True
    while run:
        batch = []
        for _ in range(BATCH_SIZE):
            doc = docs.get()
            if doc is None:
                run = False
                break
            batch.append(doc)
        if not batch:
            break
        client.session.add_all([Document(
            id=doc.id,
            title=doc.title,
            url=doc.url,
            file_path=doc.file_path,
        ) for doc in batch])
        client.session.commit()

def chunk_consumer(chunks: Queue[ChunkDatapoint], client: Client):
    run = True
    while run:
        batch = []
        for _ in range(BATCH_SIZE):
            chunk = chunks.get()
            if chunk is None:
                run = False
                break
            batch.append(chunk)
        if not batch:
            break
        client.session.add_all([Chunk(
            doc_id=point.doc_id,
            content=point.text
        ) for point in batch])
        client.session.commit()


def preprocess(root: Path, encoder: Encoder):
    logger.info(f"Starting preprocessing of Wiki dataset at {root}")
    wset = WikiDataset(root, encoder=encoder)
    logger.info(f"Loaded {len(wset.files)} files with {len(wset.docs)} documents and {len(wset.datapoints)} chunks.")
    with Client(DB_URL) as client:
        # upload files
        logger.info(f"Uploading {len(wset.files)} files to the database.")
        client.session.add_all([FileRecord(
            path=fpath.path
        ) for fpath in wset.files.values()])
        client.session.commit()
        # upload documents
        logger.info(f"Uploading {len(wset.docs)} documents to the database.")
        client.session.add_all([Document(
            id=doc.id,
            title=doc.title,
            url=doc.url,
            file_path=doc.file_path,
        ) for doc in wset.docs.values()])
        client.session.commit()
        # update links
        logger.info(f"Updating document links in the database.")
        for doc in wset.docs.values():
            db_doc = client.session.get(Document, doc.id)
            assert db_doc, f"Document with id {doc.id} not found in DB."
            linked_docs = [
                linked for lid in doc.links
                if (linked := client.session.query(Document).filter(
                    Document.title == lid
                ).first()) is not None
            ]
            db_doc.links = linked_docs
        client.session.commit()
        # upload chunks
        logger.info(f"Uploading {len(wset.datapoints)} chunks to the database.")
        client.session.add_all([Chunk(
            doc_id=point.id,
            content=point.text
        ) for point in wset.datapoints.values()])
        client.session.commit()

if __name__ == "__main__":
    # root = Path("/home/andrew/Documents/wikiparse/extracted")
    # wset = WikiDataset(root, encoder=ColBert())

    # wset.save(Path("./output"))
    
    # example_doc = wset.files["AA/wiki_00"].docs["Wallacea angulicollis"]
    # print(example_doc.chunks)
    # print(f"Loaded {len(wset.files)} files.")

    # # Inspect a file
    # for file_key, wfile in list(wset.files.items())[666:667]:
    #     print(f"File: {file_key} has {len(wfile.docs)} documents.")
    #     doc = next(iter(wfile.docs.values()))
    #     print(f"Document: {doc.title} has chunks:")
    #     for chunk in doc.chunks:
    #         print(f"- {chunk}")
    
    preprocess(Path("/home/andrew/Documents/wikiparse/extracted_full"), encoder=ColBert())