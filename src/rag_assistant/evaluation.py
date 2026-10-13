"""Retrieval evaluation: Hit@k, Recall@k and Mean Reciprocal Rank at the source-document level.

Eval file format (JSON list):
    [{"question": "...", "relevant_sources": ["paper_a.txt"]}, ...]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import ResearchAssistant


@dataclass
class EvalReport:
    n_questions: int
    k: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    failures: list[dict]

    def format(self) -> str:
        lines = [
            f"Questions evaluated : {self.n_questions}",
            f"Hit@{self.k:<2}             : {self.hit_at_k:.3f}",
            f"Recall@{self.k:<2}          : {self.recall_at_k:.3f}",
            f"MRR                 : {self.mrr:.3f}",
        ]
        if self.failures:
            lines.append("Missed questions:")
            lines += [f"  - {f['question']} (expected {f['expected']}, got {f['retrieved']})" for f in self.failures]
        return "\n".join(lines)


def evaluate_retrieval(assistant: ResearchAssistant, eval_path: str | Path, k: int = 5, mode: str = "hybrid") -> EvalReport:
    items = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    hits = recalls = rr = 0.0
    failures: list[dict] = []
    for item in items:
        expected = set(item["relevant_sources"])
        results = assistant.search(item["question"], top_k=k, mode=mode)
        retrieved = [r.chunk.source for r in results]
        found = expected & set(retrieved)
        hits += bool(found)
        recalls += len(found) / len(expected)
        first = next((rank for rank, src in enumerate(retrieved, 1) if src in expected), None)
        rr += 1 / first if first else 0.0
        if not found:
            failures.append({"question": item["question"], "expected": sorted(expected), "retrieved": retrieved})
    n = max(len(items), 1)
    return EvalReport(len(items), k, hits / n, recalls / n, rr / n, failures)
