"""High-level API tying ingestion, retrieval and generation together."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .config import Settings
from .embeddings import get_embedder
from .generation import Generator, get_generator
from .ingestion import chunk_documents, discover_files, load_document
from .retrieval import HybridRetriever, RetrievedChunk
from .vectorstore import KnowledgeBase


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    def format(self) -> str:
        lines = [self.text, "", "Sources:"]
        for i, r in enumerate(self.sources, start=1):
            lines.append(f"  [{i}] {r.chunk.source} (page {r.chunk.page}) - relevance {r.score:.4f}")
        return "\n".join(lines)


class ResearchAssistant:
    """Ingest papers, then ask grounded, cited questions about them."""

    def __init__(self, settings: Optional[Settings] = None, generator: Optional[Generator] = None) -> None:
        self.settings = settings or Settings()
        self._generator = generator
        if KnowledgeBase.exists(self.settings.index_dir):
            self.kb = KnowledgeBase.load(self.settings.index_dir)
        else:
            embedder = get_embedder(self.settings.embedder, self.settings.embedding_model)
            self.kb = KnowledgeBase(embedder, self.settings.embedder, self.settings.embedding_model)
        self._retriever: Optional[HybridRetriever] = None

    # ------------------------------------------------------------------ properties
    @property
    def generator(self) -> Generator:
        if self._generator is None:  # created lazily so ingestion never needs API keys
            self._generator = get_generator(self.settings)
        return self._generator

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                self.kb, self.settings.dense_weight, self.settings.bm25_weight, self.settings.rrf_k
            )
        return self._retriever

    # ------------------------------------------------------------------ ingestion
    def ingest(self, paths: Iterable[str | Path], persist: bool = True) -> dict:
        """Index files/directories. Returns a summary of what was added."""
        files = discover_files(paths)
        added = 0
        for file in files:
            docs = load_document(file)
            chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
            added += self.kb.add_chunks(chunks)
        self._retriever = None
        if persist and files:
            self.kb.save(self.settings.index_dir)
        return {"files": len(files), "new_chunks": added, "total_chunks": len(self.kb)}

    def remove(self, source: str, persist: bool = True) -> int:
        removed = self.kb.remove_source(source)
        self._retriever = None
        if persist:
            self.kb.save(self.settings.index_dir)
        return removed

    # ------------------------------------------------------------------ querying
    def search(self, query: str, top_k: Optional[int] = None, mode: str = "hybrid") -> list[RetrievedChunk]:
        return self.retriever.search(query, top_k or self.settings.top_k, mode)

    def ask(self, question: str, top_k: Optional[int] = None, mode: str = "hybrid") -> Answer:
        results = self.search(question, top_k, mode)
        return Answer(question, self.generator.generate(question, results), results)

    def stats(self) -> dict:
        return {
            "sources": self.kb.sources,
            "n_sources": len(self.kb.sources),
            "n_chunks": len(self.kb),
            "embedder": self.kb.embedder_kind,
            "llm_backend": self.settings.llm_backend,
        }
