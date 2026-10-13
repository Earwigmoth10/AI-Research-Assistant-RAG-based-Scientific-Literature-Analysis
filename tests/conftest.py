from pathlib import Path

import pytest

from rag_assistant.config import Settings
from rag_assistant.pipeline import ResearchAssistant

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_papers"
EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval" / "sample_eval.json"


@pytest.fixture()
def assistant(tmp_path):
    settings = Settings(index_dir=tmp_path / "index", chunk_size=60, chunk_overlap=15)
    bot = ResearchAssistant(settings)
    bot.ingest([SAMPLES])
    return bot
