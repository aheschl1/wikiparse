import pytest
from wikindex.wiki.wiki import WikiDoc
from wikindex.config import Config
from wikindex.custom_colbert.model import ColBert

@pytest.fixture(scope="module")
def encoder():
    return ColBert()

def test_short_content_fits_in_one_chunk(encoder):
    content = "This is a short paragraph."
    config = Config(max_tokens=32)  # artificially small
    chunks = WikiDoc.chunk_content(content, title="Title", encoder=encoder, config=config)
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]

def test_long_content_splits_into_chunks(encoder):
    content = "Sentence one. Sentence two. " * 50
    config = Config(max_tokens=32)
    chunks = WikiDoc.chunk_content(content, title="Title", encoder=encoder, config=config)
    assert len(chunks) > 1
    # Every chunk should be <= max_tokens
    for chunk in chunks:
        tokens = encoder.tokenizer.tokenize(chunk)
        assert len(tokens) <= config.max_tokens - 2
