import pytest
from pathlib import Path
from wikindex.wiki.wiki import WikiFile, WikiDataset
from wikindex.custom_colbert.model import ColBertV2
from wikindex.config import Config

@pytest.fixture
def tmp_wiki_file(tmp_path: Path):
    """Create a fake wiki file with two docs."""
    content = """
    <doc id="1" url="https://example.org/1" title="First">
    First paragraph.

    Second paragraph with a <a href="LinkTitle">link</a>.
    </doc>
    <doc id="2" url="https://example.org/2" title="Second">
    Another document content.
    </doc>
    """
    f = tmp_path / "wiki_00"
    f.write_text(content)
    return f

def test_parse_file(tmp_wiki_file):
    wf = WikiFile(tmp_wiki_file, encoder=ColBertV2(), config=Config(max_tokens=32))
    assert "First" in wf.docs
    assert "Second" in wf.docs

    first_doc = wf.docs["First"]
    assert isinstance(first_doc.chunks, list)
    assert "Second paragraph with a link." in " ".join(first_doc.chunks)
    assert "LinkTitle" in first_doc.links

def test_wiki_set(tmp_path: Path, tmp_wiki_file):
    # put wiki_00 under AA/ to match expected path structure
    subdir = tmp_path / "AA"
    subdir.mkdir()
    (subdir / "wiki_00").write_text(tmp_wiki_file.read_text())

    ws = WikiDataset(tmp_path, encoder=ColBertV2(), config=Config(max_tokens=32))
    assert "First" in ws.docs
    assert "Second" in ws.docs
