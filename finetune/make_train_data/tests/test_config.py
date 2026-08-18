from make_train_data.config import config


def test_기본값들이_로드된다():
    assert config.MTD_NARROW_WINDOW_HOURS == 72
    assert config.MTD_BROAD_WINDOW_DAYS == 28
    assert config.MTD_ENTITY_JACCARD_THRESHOLD == 0.3
    assert config.MTD_EMBED_SIM_THRESHOLD == 0.75
    assert config.MTD_DEDUP_THRESHOLD == 0.9
    assert config.MTD_MIN_CLUSTER_SIZE == 2
    assert config.MTD_ONEFACT_RATIO == 0.175
    assert config.MTD_MIN_ARTICLES == 20


def test_타입이_숫자다():
    assert isinstance(config.MTD_NARROW_WINDOW_HOURS, float)
    assert isinstance(config.MTD_MIN_CLUSTER_SIZE, int)
    assert isinstance(config.MTD_MIN_ARTICLES, int)
