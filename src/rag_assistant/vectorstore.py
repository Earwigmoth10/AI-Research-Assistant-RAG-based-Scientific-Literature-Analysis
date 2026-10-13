"""A small persistent knowledge base: chunks + their vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import Embedder, get_embedder
from .ingestion import Chunk


class KnowledgeBase:
    """Holds chunks and the embedding matrix; saves to / loads from a directory."""

    def __init__(self, embedder: Embedder, embedder_kind: str = "tfidf", model_name: str = "") -> None:
        self.embedder = embedder
        self.embedder_kind = embedder_kind
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.matrix = None  # dense ndarray or scipy sparse matrix

    # ------------------------------------------------------------------ building
    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Add new chunks (duplicates by id are skipped) and rebuild the index."""
        known = {c.id for c in self.chunks}
        new = [c for c in chunks if c.id not in known]
        if not new:
            return 0
        self.chunks.extend(new)
        self.rebuild()
        return len(new)

    def rebuild(self) -> None:
        texts = [c.text for c in self.chunks]
        if not texts:
            self.matrix = None
            return
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)

    def remove_source(self, source: str) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        self.rebuild()
        return before - len(self.chunks)

    # ------------------------------------------------------------------ search
    def dense_scores(self, query: str) -> np.ndarray:
        """Cosine similarity between the query and every chunk."""
        if self.matrix is None or not self.chunks:
            return np.zeros(0)
        q = self.embedder.encode([query])
        scores = self.matrix @ q.T
        if hasattr(scores, "toarray"):
            scores = scores.toarray()
        return np.asarray(scores).ravel()

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        meta = {"embedder": self.embedder_kind, "model_name": self.model_name, "n_chunks": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        vectors = directory / "vectors.npy"
        if self.embedder.persist_vectors and self.matrix is not None:
            np.save(vectors, self.matrix)
        elif vectors.exists():
            vectors.unlink()

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeBase":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        embedder = get_embedder(meta["embedder"], meta.get("model_name") or "")
        kb = cls(embedder, meta["embedder"], meta.get("model_name", ""))
        kb.chunks = chunks
        vectors = directory / "vectors.npy"
        if embedder.persist_vectors and vectors.exists():
            kb.matrix = np.load(vectors)
        else:
            kb.rebuild()
        return kb

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "chunks.json").exists() and (directory / "meta.json").exists()
