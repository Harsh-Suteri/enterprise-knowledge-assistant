"""Tests for the parts that don't need a model or network.

Chunking is the highest-leverage thing to test: a silent regression here
degrades every answer downstream, and it's cheap to verify.
"""

import pytest

from app.config import settings
from app.ingest import Chunk, split


def test_split_produces_chunks():
    text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
    chunks = split(text, source="test.md")
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "test.md" for c in chunks)


def test_chunk_ids_are_unique_and_ordered():
    text = " ".join(f"sentence number {i}." for i in range(400))
    chunks = split(text, source="doc.txt")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique or upsert will collide"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_limit():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split(text, source="doc.txt")
    # The splitter can overshoot slightly when a single token exceeds the
    # limit, so allow a small margin rather than asserting a hard ceiling.
    assert all(len(c.text) <= settings.chunk_size * 1.3 for c in chunks)


def test_empty_text_produces_no_chunks():
    assert split("   \n\n  ", source="blank.txt") == []


def test_overlap_preserves_boundary_content():
    """A sentence straddling a boundary should survive in at least one chunk."""
    marker = "THISEXACTPHRASEMUSTSURVIVE"
    filler = "padding text. " * 40
    text = f"{filler}{marker}. {filler}"
    chunks = split(text, source="doc.txt")
    assert any(marker in c.text for c in chunks)


@pytest.mark.parametrize("suffix", [".pdf", ".txt", ".md", ".docx"])
def test_supported_extensions_declared(suffix):
    from app.ingest import SUPPORTED

    assert suffix in SUPPORTED
