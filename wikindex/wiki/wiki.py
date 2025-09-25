from pathlib import Path
from re import compile
import re
from pydantic import BaseModel, Field
import html
from glob import glob

from wikindex.config import Config
from wikindex.custom_colbert.model import ColBertV2, Encoder

import logging

from wikindex.data.dataset import Datapoint, Dataset

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class ChunkDatapoint(Datapoint, BaseModel):
    text: str
    id: int
    doc_title: str
    
    @property
    def metadata(self) -> dict:
        return {"doc_title": self.doc_title}

class WikiDoc(BaseModel):
    id: int = Field(..., description="The unique identifier for the document.")
    url: str = Field(..., description="The URL of the document.")
    title: str = Field(..., description="The title of the document.")
    chunks: list[str] = Field(..., description="The main content of the document, split into chunks.")
    links: list[str] = Field(default_factory=list, description="List of titles of referenced docs.")

    def get_datapoints(self, ids: list[int]):
        for i, chunk in enumerate(self.chunks):
            yield ChunkDatapoint(text=chunk, doc_title=self.title, id=ids[i])

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
        self.path = path
        self.config = config
        self.encoder = encoder
        self.docs: dict[str, WikiDoc] = self._parse()

    def _load_text(self) -> str:
        return self.path.read_text(encoding='utf-8')

    def _parse(self) -> dict[str, WikiDoc]:
        docs = {}
        for match in DOC_PATTERN.finditer(self._load_text()):
            doc_id, url, title, raw_content = (
                int(match["id"]),
                match["url"],
                match["title"],
                match["content"].strip(),
            )
            content, references = clean_content(raw_content)
            docs[title] = WikiDoc(
                id=doc_id,
                url=url,
                title=title,
                chunks=WikiDoc.chunk_content(content, title, self.encoder, self.config),
                links=references,
            )
        return docs

class WikiDataset(Dataset):
    def __init__(self, root: Path, encoder: Encoder, config: Config = Config()):
        self.root = root
        self.config = config
        self.encoder = encoder
        self._datapoints = None

        self.files: dict[str, WikiFile] = self._load_files()
        self.docs: dict[str, WikiDoc] = self._collect_docs()

    def _load_files(self) -> dict[str, WikiFile]:
        files = {}
        for path in map(Path, glob(str(self.root / "*" / "*_*"))):
            files[f"{path.parent.name}/{path.name}"] = WikiFile(
                path, encoder=self.encoder, config=self.config
            )
        return files
    
    def _collect_docs(self) -> dict[str, WikiDoc]:
        return {title: doc for wf in self.files.values() for title, doc in wf.docs.items()}
    
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

if __name__ == "__main__":
    root = Path("/home/andrew/Documents/wikiparse/extracted")
    wset = WikiDataset(root, encoder=ColBertV2())

    example_doc = wset.files["AA/wiki_00"].docs["Wallacea angulicollis"]
    print(example_doc.chunks)
    print(f"Loaded {len(wset.files)} files.")

    # Inspect a file
    for file_key, wfile in list(wset.files.items())[666:667]:
        print(f"File: {file_key} has {len(wfile.docs)} documents.")
        doc = next(iter(wfile.docs.values()))
        print(f"Document: {doc.title} has chunks:")
        for chunk in doc.chunks:
            print(f"- {chunk}")