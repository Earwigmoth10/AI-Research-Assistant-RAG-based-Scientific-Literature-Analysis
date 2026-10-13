"""Answer generation. Every back-end receives the same retrieved, numbered context."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from .config import Settings
from .retrieval import RetrievedChunk, tokenize

SYSTEM_PROMPT = (
    "You are a careful research assistant that answers questions about scientific literature. "
    "Use ONLY the numbered context passages provided. Cite the passages you rely on with bracketed "
    "numbers such as [1] or [2][3]. If the context does not contain the answer, say that the "
    "provided papers do not cover it instead of guessing. Be concise and precise, and keep the "
    "terminology of the source papers."
)

NO_CONTEXT_MESSAGE = "I could not find relevant passages in the indexed papers for this question."


def build_context(results: list[RetrievedChunk]) -> str:
    blocks = [f"[{i}] (source: {r.chunk.source}, page {r.chunk.page})\n{r.chunk.text}" for i, r in enumerate(results, 1)]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[RetrievedChunk]) -> str:
    return f"Context passages:\n\n{build_context(results)}\n\nQuestion: {question}\n\nAnswer with citations:"


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, results: list[RetrievedChunk]) -> str: ...


class ExtractiveGenerator(Generator):
    """Offline fallback: returns the most question-relevant sentences with citations.

    It does not paraphrase or reason, but it is deterministic, free and always grounded.
    """

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_MESSAGE
        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, int, str]] = []
        for ref, res in enumerate(results, start=1):
            for pos, sent in enumerate(re.split(r"(?<=[.!?])\s+", res.chunk.text)):
                overlap = len(q_terms & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap / (1 + 0.02 * len(sent.split())), ref, pos, sent.strip()))
        if not scored:
            return NO_CONTEXT_MESSAGE
        top = sorted(scored, key=lambda s: s[0], reverse=True)[: self.max_sentences]
        top.sort(key=lambda s: (s[1], s[2]))  # present in reading order
        body = " ".join(f"{sent} [{ref}]" for _, ref, _, sent in top)
        return f"Most relevant evidence from the indexed papers: {body}"


class AnthropicGenerator(Generator):
    def __init__(self, settings: Settings) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'ai-research-assistant-rag[anthropic]'") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.settings = settings

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_MESSAGE
        response = self.client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, results)}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


class OpenAIGenerator(Generator):
    def __init__(self, settings: Settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'ai-research-assistant-rag[openai]'") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.settings = settings

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_MESSAGE
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, results)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def get_generator(settings: Settings) -> Generator:
    backend = settings.llm_backend.lower()
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "anthropic":
        return AnthropicGenerator(settings)
    if backend == "openai":
        return OpenAIGenerator(settings)
    raise ValueError(f"Unknown LLM backend '{backend}'. Use 'extractive', 'anthropic' or 'openai'.")
