from conftest import SAMPLES

from rag_assistant.ingestion import Document, chunk_document, clean_text, discover_files, load_document


def test_clean_text_rejoins_hyphenation_and_newlines():
    assert clean_text("retrie-\nval works\nwell") == "retrieval works well"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_document(Document("a.txt", 1, text), chunk_size=40, overlap=10)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 40 for c in chunks)
    # consecutive chunks share some text because of the overlap
    assert set(chunks[0].text.split()[-6:]) & set(chunks[1].text.split()[:12])


def test_chunk_ids_are_stable_and_unique():
    doc = Document("a.txt", 1, "One sentence here. Another sentence there. " * 30)
    first = chunk_document(doc, 20, 5)
    second = chunk_document(doc, 20, 5)
    assert [c.id for c in first] == [c.id for c in second]


def test_load_and_discover_text_files():
    files = discover_files([SAMPLES])
    assert len(files) == 3
    docs = load_document(files[0])
    assert docs and docs[0].source == files[0].name
