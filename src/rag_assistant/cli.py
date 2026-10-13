"""Command line interface: ``rag-assistant ingest | ask | search | eval | stats``."""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .evaluation import evaluate_retrieval
from .pipeline import ResearchAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-assistant", description="RAG-based scientific literature assistant")
    parser.add_argument("--index-dir", help="where the index is stored (default: ./index)")
    parser.add_argument("--embedder", choices=["tfidf", "sentence-transformers"], help="embedding back-end")
    parser.add_argument("--llm", choices=["extractive", "anthropic", "openai"], help="answer generation back-end")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index PDFs / .txt / .md files or folders")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("ask", help="ask a question and get a cited answer")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")

    p = sub.add_parser("search", help="show retrieved passages without generating an answer")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")

    p = sub.add_parser("eval", help="evaluate retrieval quality on a labelled question file")
    p.add_argument("eval_file")
    p.add_argument("-k", "--top-k", type=int, default=3)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")

    sub.add_parser("stats", help="show what is currently indexed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.index_dir:
        settings.index_dir = type(settings.index_dir)(args.index_dir)
    if args.embedder:
        settings.embedder = args.embedder
    if args.llm:
        settings.llm_backend = args.llm
    assistant = ResearchAssistant(settings)

    if args.command == "ingest":
        summary = assistant.ingest(args.paths)
        print(f"Indexed {summary['files']} file(s): +{summary['new_chunks']} new chunks, {summary['total_chunks']} total.")
        return 0 if summary["files"] else 1

    if len(assistant.kb) == 0 and args.command != "stats":
        print("The index is empty. Run `rag-assistant ingest <files-or-folders>` first.", file=sys.stderr)
        return 1

    if args.command == "ask":
        print(assistant.ask(args.question, args.top_k, args.mode).format())
    elif args.command == "search":
        for i, r in enumerate(assistant.search(args.query, args.top_k, args.mode), start=1):
            print(f"[{i}] {r.chunk.source} p.{r.chunk.page}  score={r.score:.4f}  (dense#{r.dense_rank}, bm25#{r.bm25_rank})")
            print(f"    {r.chunk.text[:300]}{'...' if len(r.chunk.text) > 300 else ''}\n")
    elif args.command == "eval":
        print(evaluate_retrieval(assistant, args.eval_file, args.top_k, args.mode).format())
    elif args.command == "stats":
        for key, value in assistant.stats().items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
