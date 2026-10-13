"""Hybrid retrieval: BM25 (lexical) + embeddings (semantic) fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ingestion import Chunk
from .vectorstore import KnowledgeBase

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was were
    which with what how why when where who whom do does did can could should would not than then
    into over using used use we our their these those been being also such between about""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of tokenised documents."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in corpus]
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=float)
        self.n = len(corpus)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        if not self.n or not self.avgdl:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([counts.get(term, 0) for counts in self.tf], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


def _ranking(scores: np.ndarray, limit: int) -> list[int]:
    """Indices of the best `limit` chunks with a strictly positive score."""
    if scores.size == 0:
        return []
    order = np.argsort(-scores)[:limit]
    return [int(i) for i in order if scores[i] > 0]


class HybridRetriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.kb = kb
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25([tokenize(c.text) for c in kb.chunks]) if kb.chunks else None

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[RetrievedChunk]:
        if not self.kb.chunks:
            return []
        pool = max(top_k * 4, 20)
        dense_rank: dict[int, int] = {}
        bm25_rank: dict[int, int] = {}

        if mode in {"hybrid", "dense"}:
            dense_rank = {i: r for r, i in enumerate(_ranking(self.kb.dense_scores(query), pool), start=1)}
        if mode in {"hybrid", "bm25"} and self.bm25 is not None:
            bm25_rank = {i: r for r, i in enumerate(_ranking(self.bm25.scores(tokenize(query)), pool), start=1)}
        if mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError("mode must be 'hybrid', 'dense' or 'bm25'")

        fused: dict[int, float] = {}
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.dense_weight / (self.rrf_k + rank)
        for idx, rank in bm25_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(self.kb.chunks[i], score, dense_rank.get(i), bm25_rank.get(i)) for i, score in best
        ]
