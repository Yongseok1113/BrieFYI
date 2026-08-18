# tests/test_config_rag_summarize.py
from config import config


def test_groq_설정_기본값():
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"


def test_rag_summarize_설정_기본값():
    assert config.RAG_SUMMARIZE_PROVIDER == "groq"
    assert config.RAG_SUMMARIZE_MAX_ATTEMPTS == 3
    assert config.RAG_SUMMARIZE_SCORE_THRESHOLD == 70.0
