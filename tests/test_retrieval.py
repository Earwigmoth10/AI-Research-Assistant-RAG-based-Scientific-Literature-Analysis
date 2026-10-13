from rag_assistant.retrieval import BM25, tokenize


def test_bm25_ranks_matching_document_first():
    corpus = [tokenize(t) for t in ["cats purr softly", "dogs bark loudly", "stuttering detection with cnn"]]
    scores = BM25(corpus).scores(tokenize("cnn stuttering"))
    assert scores.argmax() == 2


def test_hybrid_finds_right_source(assistant):
    results = assistant.search("positional encodings in attention", top_k=3)
    assert results[0].chunk.source == "transformer_architecture.txt"


def test_modes_return_results(assistant):
    for mode in ("hybrid", "dense", "bm25"):
        assert assistant.search("speaker independent stuttering", top_k=2, mode=mode)


def test_unknown_query_returns_nothing(assistant):
    assert assistant.search("qwertyuiop zxcvbnm", top_k=3) == []
