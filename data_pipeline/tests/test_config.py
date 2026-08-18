from data_pipeline.config import config


def test_defaults_have_expected_types():
    assert isinstance(config.RATE_LIMIT_MAX_REQUESTS, int)
    assert isinstance(config.RATE_LIMIT_WINDOW_SECONDS, float)
    assert isinstance(config.SYNONYM_CLUSTER_THRESHOLD, float)
    assert isinstance(config.SYNONYM_FUZZY_THRESHOLD, float)
    assert isinstance(config.BATCH_LIMIT, int)


def test_llm_provider_defaults_to_groq():
    assert config.LLM_PROVIDER == "groq"


def test_database_url_is_constructed_when_not_set_explicitly():
    assert config.DATABASE_URL.startswith("postgresql://")
