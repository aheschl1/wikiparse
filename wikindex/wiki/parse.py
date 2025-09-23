from pathlib import Path
from re import compile
import re
from pydantic import BaseModel, Field
import html
from glob import glob

from wikindex.config import Config
from wikindex.custom_colbert.model import ColBertV2, Encoder

import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class WikiDoc(BaseModel):
    id: int = Field(..., description="The unique identifier for the document.")
    url: str = Field(..., description="The URL of the document.")
    title: str = Field(..., description="The title of the document.")
    chunks: list[str] = Field(..., description="The main content of the document, split into chunks.")
    links: list[str] = Field(default_factory=list, description="List of titles of referenced docs.")

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
        initial_chunks = [
            chunk.strip()
            for chunk in clean_content.split("\n\n")
            if chunk.strip() and chunk.strip() != title
        ]

        final_chunks: list[str] = []

        # Stage 2: process each chunk
        for chunk in initial_chunks:
            tokens = tokenizer.tokenize(chunk)

            if len(tokens) <= max_tokens - 2:
                # already fits, keep it
                final_chunks.append(chunk)
            else:
                # split into sentences and repack greedily
                sentences = re.split(r'(?<=[.!?]) +', chunk)
                current_tokens: list[str] = []

                for sentence in sentences:
                    sent_tokens = tokenizer.tokenize(sentence)

                    if len(current_tokens) + len(sent_tokens) <= max_tokens - 2:
                        current_tokens.extend(sent_tokens)
                    else:
                        if current_tokens:
                            # re-encode so we can get back to text
                            final_chunks.append(
                                tokenizer.convert_tokens_to_string(current_tokens)
                            )
                        current_tokens = sent_tokens

                if current_tokens:
                    final_chunks.append(
                        tokenizer.convert_tokens_to_string(current_tokens)
                    )

        return final_chunks

class WikiFile:
    def __init__(self, path: Path, encoder: Encoder, config: Config = Config()):
        self.path = path
        self.config = config
        self.encoder = encoder
        self.docs = self.parse()

    def _load_text(self) -> str:
        with open(self.path, "r") as f:
            return f.read()
        
    def _clean_content(self, raw_content: str):
        """
        Given raw content like:
        
        Wallacea angulicollis is a species of &lt;a href="beetle"&gt;beetle&lt;/a&gt; in the family &lt;a href="Chrysomelidae"&gt;Chrysomelidae&lt;/a&gt;. It is found in Malaysia (Sarawak).
        
        Cleans up html, and returns the cleaned content along with a list of references (the hrefs)
        """
        # html unescape
        content = html.unescape(raw_content)
        # find all links, pull the href, and replace the full <a> tag with just the link text
        references = []
        def replace_link(match):
            href = match.group(1)
            link_text = match.group(2)
            references.append(href)
            return link_text
        content = re.sub(r'<a href="([^"]+)">(.*?)</a>', replace_link, content)
        # remove any other html tags
        content = re.sub(r'<[^>]+>', '', content)
        return content, references
    
    def parse(self):
        reg = compile(
            r'<doc id="(?P<id>\d+)" url="(?P<url>[^"]+)" title="(?P<title>[^"]+)">(?P<content>.*?)</doc>', re.DOTALL
        )
        matches = reg.finditer(self._load_text())
        docs = {}
        for match in matches:
            title = match.group("title")
            # clean content a bit. extract links, and put only the link text
            content, references = self._clean_content(match.group("content").strip())
            doc = WikiDoc(
                id=int(match.group("id")),
                url=match.group("url"),
                title=title,
                chunks=WikiDoc.chunk_content(content, title, self.encoder, self.config),
                links=references,
            )
            docs[doc.title] = doc
        return docs

class WikiSet:
    def __init__(self, root: Path, encoder: Encoder, config: Config = Config()):
        self.root = root
        self.config = config
        self.encoder = encoder
        self.paths = glob(str(root / "*" / "*_*"))
        self.files = self.load_files()
    
    def load_files(self):
        files = {}
        for path in self.paths:
            p = Path(path)
            wf = WikiFile(p, config=self.config, encoder=self.encoder)
            files[f"{p.parent.name}/{p.name}"] = wf
        return files
    
    def chunks(self):
        for file_key, wfile in self.files.items():
            for doc_key, doc in wfile.docs.items():
                for chunk in doc.chunks:
                    yield (file_key, doc_key, chunk)
        
if __name__ == "__main__":
    root = Path("/home/andrew/Documents/wikiparse/extracted")
    wset = WikiSet(root, encoder=ColBertV2())
    print(wset.files["AA/wiki_00"].docs["Wallacea angulicollis"].chunks)
    print(f"Loaded {len(wset.files)} files.")
    # check chunks
    config = Config()
    i = 0
    for file_key, wfile in wset.files.items():
        if i < 666:
            i += 1
            continue
        i += 1
        print(f"File: {file_key} has {len(wfile.docs)} documents.")
        for doc_key, doc in wfile.docs.items():
            print(f"Document: {doc_key} has chunks:")
            for chunk in doc.chunks:
                print(f"- {chunk}")
            break  # just show one document per file
        break  # just show one file