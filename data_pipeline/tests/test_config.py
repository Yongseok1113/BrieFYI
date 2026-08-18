import importlib
import os

from data_pipeline.config import config


def test_defaults_have_expected_types():
    assert isinstance(config.RATE_LIMIT_MAX_REQUESTS, int)
    assert isinstance(config.RATE_LIMIT_WINDOW_SECONDS, float)
    assert isinstance(config.SYNONYM_CLUSTER_THRESHOLD, float)
    assert isinstance(config.SYNONYM_FUZZY_THRESHOLD, float)
    assert isinstance(config.BATCH_LIMIT, int)


def test_llm_provider_defaults_to_groq(monkeypatch):
    """DATA_PIPELINE_LLM_PROVIDER가 설정 안 됐을 때만 검증한다.

    config 모듈은 import 시점에 실제 루트 .env를 읽는 싱글턴이라, 로컬 .env에
    DATA_PIPELINE_LLM_PROVIDER가 이미 설정돼 있으면 단순히 config.LLM_PROVIDER를
    확인하는 것만으로는 "기본값"을 검증할 수 없다. 환경변수를 지우고 모듈을
    리로드해 기본값 경로를 강제로 태운 뒤, 다른 테스트에 영향이 없도록 원래
    상태로 복원한다.
    """
    from data_pipeline import config as config_module

    # Save the current environment state
    original_value = os.environ.get("DATA_PIPELINE_LLM_PROVIDER")

    # Patch dotenv.load_dotenv at the source so reload picks up the mocked version
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)

    monkeypatch.delenv("DATA_PIPELINE_LLM_PROVIDER", raising=False)
    try:
        importlib.reload(config_module)
        assert config_module.config.LLM_PROVIDER == "groq"
    finally:
        # Restore the original environment before reloading
        if original_value is not None:
            monkeypatch.setenv("DATA_PIPELINE_LLM_PROVIDER", original_value)
        else:
            monkeypatch.delenv("DATA_PIPELINE_LLM_PROVIDER", raising=False)
        importlib.reload(config_module)


def test_database_url_is_constructed_when_not_set_explicitly():
    assert config.DATABASE_URL.startswith("postgresql://")
