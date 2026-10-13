"""Embedding back-ends.

* ``tfidf``  - fully offline TF-IDF vectors (default, no model download needed)
* ``sentence-transformers`` - dense semantic embeddings (recommended for best quality)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(ABC):
    """Turns text into L2-normalised vectors so a dot product equals cosine similarity."""

    name: str = "base"

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "Embedder":
        """Learn corpus statistics (a no-op for pretrained models)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]):
        """Return a matrix (dense ndarray or scipy sparse) of shape (n, dim)."""

    @property
    def persist_vectors(self) -> bool:
        """Whether the chunk matrix should be saved to disk (expensive models) or refit on load."""
        return False


class TfidfEmbedder(Embedder):
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            norm="l2",
        )

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(list(texts))
        return self

    def encode(self, texts: Sequence[str]):
        return self.vectorizer.transform(list(texts))


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install the optional extra: pip install 'ai-research-assistant-rag[dense]'"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def persist_vectors(self) -> bool:
        return True


def get_embedder(kind: str = "tfidf", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    kind = kind.lower()
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind in {"sentence-transformers", "dense", "st"}:
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder '{kind}'. Choose 'tfidf' or 'sentence-transformers'.")
