import pytest
from wikindex.wiki.wiki import clean_content

def test_clean_content_basic():
    raw = 'A <a href="Dog">dog</a> is an <a href="Animal">animal</a>.'
    cleaned, refs = clean_content(raw)

    assert "dog" in cleaned
    assert "animal" in cleaned
    assert "<a" not in cleaned  # all links removed
    assert refs == ["Dog", "Animal"]

def test_clean_content_with_html_entities():
    raw = "5 &lt; 10 and &amp; is an ampersand"
    cleaned, refs = clean_content(raw)

    assert cleaned == "5 < 10 and & is an ampersand"
    assert refs == []
