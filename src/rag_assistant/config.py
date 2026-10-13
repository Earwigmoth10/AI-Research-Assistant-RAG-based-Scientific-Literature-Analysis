"""Central configuration, driven by environment variables with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional convenience: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings for the pipeline.

    Every field can be overridden through an environment variable (see `.env.example`).
    """

    # Storage
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_INDEX_DIR", "index")))

    # Chunking (measured in words)
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 180))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 40))

    # Retrieval
    embedder: str = field(default_factory=lambda: os.getenv("RAG_EMBEDDER", "tfidf"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    dense_weight: float = field(default_factory=lambda: _env_float("RAG_DENSE_WEIGHT", 1.0))
    bm25_weight: float = field(default_factory=lambda: _env_float("RAG_BM25_WEIGHT", 1.0))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))

    # Generation
    llm_backend: str = field(default_factory=lambda: os.getenv("RAG_LLM_BACKEND", "extractive"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_MAX_TOKENS", 800))
    temperature: float = field(default_factory=lambda: _env_float("RAG_TEMPERATURE", 0.1))

    def __post_init__(self) -> None:
        self.index_dir = Path(self.index_dir)
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
