"""Document loading and chunking.

Supports PDF, plain text and Markdown. Chunks are sentence-aware, overlap slightly
and remember which source file and page they came from so answers can cite them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    """A page (or whole file, for text formats) of raw text."""

    source: str
    page: int
    text: str


@dataclass
class Chunk:
    """A retrievable passage of a document."""

    id: str
    source: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def clean_text(text: str) -> str:
    """Normalise whitespace and repair hyphenation introduced by PDF line breaks."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> spaces
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def load_document(path: str | Path) -> list[Document]:
    """Load a file into one `Document` per page (PDF) or one for the whole file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}' for {path.name}")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PDF support requires `pypdf` (pip install pypdf)") from exc
        reader = PdfReader(str(path))
        docs = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                docs.append(Document(source=path.name, page=number, text=text))
        return docs

    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(source=path.name, page=1, text=text)] if text else []


def discover_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of supported files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(p)
    return found


def _chunk_id(source: str, page: int, text: str) -> str:
    return hashlib.sha1(f"{source}|{page}|{text}".encode("utf-8")).hexdigest()[:16]


def chunk_document(doc: Document, chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Split one document into overlapping, sentence-aligned chunks (sizes in words)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Break very long "sentences" (tables, reference lists) into word windows.
    sentences: list[str] = []
    for sent in split_sentences(doc.text):
        words = sent.split()
        if len(words) <= chunk_size:
            sentences.append(sent)
        else:
            for i in range(0, len(words), chunk_size):
                sentences.append(" ".join(words[i : i + chunk_size]))

    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        text = " ".join(current).strip()
        if text:
            chunks.append(Chunk(_chunk_id(doc.source, doc.page, text), doc.source, doc.page, text))

    for sent in sentences:
        n = len(sent.split())
        if current and count + n > chunk_size:
            flush()
            # carry over trailing sentences as overlap
            carried: list[str] = []
            carried_words = 0
            for prev in reversed(current):
                w = len(prev.split())
                if carried_words + w > overlap:
                    break
                carried.insert(0, prev)
                carried_words += w
            current, count = carried, carried_words
        current.append(sent)
        count += n
    if current:
        flush()
    return chunks


def chunk_documents(docs: Iterable[Document], chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size, overlap))
    return out
