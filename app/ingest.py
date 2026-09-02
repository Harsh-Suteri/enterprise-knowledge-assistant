"""Load documents from disk and split them into retrievable chunks.

The two decisions that matter here are (1) how a file becomes text and
(2) where that text gets cut. Everything downstream inherits those choices,
so both are explicit and configurable rather than buried in a library call.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class Chunk:
    """One retrievable unit of text, plus where it came from.

    `source` and `chunk_index` are what let an answer cite its evidence —
    without them the system can retrieve but can't attribute.
    """

    text: str
    source: str
    chunk_index: int

    @property
    def id(self) -> str:
        return f"{self.source}::{self.chunk_index}"


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Pages that fail to extract return None; skip rather than crash the run.
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def load_text(path: Path) -> str:
    """Turn a single file into plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def split(text: str, source: str) -> list[Chunk]:
    """Split text on natural boundaries, largest first.

    RecursiveCharacterTextSplitter tries paragraph breaks, then line breaks,
    then sentences, then words — only falling back to a hard character cut
    when nothing else fits. That keeps most chunks semantically whole, which
    is what makes the embedding meaningful.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [
        Chunk(text=piece, source=source, chunk_index=i)
        for i, piece in enumerate(splitter.split_text(text))
        if piece.strip()
    ]


def ingest_file(path: Path) -> list[Chunk]:
    return split(load_text(path), source=path.name)


def ingest_dir(directory: Path | None = None) -> list[Chunk]:
    """Ingest every supported file in a directory."""
    directory = directory or settings.data_dir
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            chunks.extend(ingest_file(path))
    return chunks
