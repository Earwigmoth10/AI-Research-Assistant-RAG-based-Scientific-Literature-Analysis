from conftest import EVAL_FILE, SAMPLES

from rag_assistant.config import Settings
from rag_assistant.evaluation import evaluate_retrieval
from rag_assistant.pipeline import ResearchAssistant


def test_ask_returns_cited_grounded_answer(assistant):
    answer = assistant.ask("Why do random window-level splits inflate accuracy?")
    assert answer.sources and "[" in answer.text
    assert answer.sources[0].chunk.source == "cnn_stuttering_detection.txt"


def test_ask_with_no_match_says_so(assistant):
    assert "could not find" in assistant.ask("qwertyuiop zxcvbnm").text.lower()


def test_ingest_is_idempotent_and_persistent(tmp_path):
    settings = Settings(index_dir=tmp_path / "idx", chunk_size=60, chunk_overlap=15)
    first = ResearchAssistant(settings)
    added = first.ingest([SAMPLES])["new_chunks"]
    assert added > 0
    assert first.ingest([SAMPLES])["new_chunks"] == 0
    reloaded = ResearchAssistant(settings)
    assert len(reloaded.kb) == len(first.kb)
    assert reloaded.search("hybrid retrieval BM25", top_k=1)[0].chunk.source == "rag_overview.txt"


def test_remove_source(assistant):
    assert assistant.remove("rag_overview.txt") > 0
    assert "rag_overview.txt" not in assistant.kb.sources


def test_retrieval_evaluation_on_sample_set(assistant):
    report = evaluate_retrieval(assistant, EVAL_FILE, k=3)
    assert report.hit_at_k >= 0.8 and report.mrr >= 0.7
