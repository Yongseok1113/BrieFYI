from datetime import datetime, timedelta

from make_train_data.clustering import cluster_articles


def _article(id_, title, published_at, entity=None):
    return {"id": id_, "title": title, "description": "", "published_at": published_at, "entity": entity, "event": None}


def _entity_fn(article):
    return (article.get("entity") or [], None, [])


def _make_embed_fn(vectors: dict[str, list[float]]):
    """title -> 고정 벡터 매핑을 쓰는 가짜 embed_fn (data_pipeline 테스트 스타일)."""

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [vectors[t.strip()] for t in texts]

    return embed_fn


BASE_TIME = datetime(2026, 8, 1, 12, 0, 0)
DEFAULT_KWARGS = dict(
    narrow_window_hours=72,
    broad_window_days=28,
    entity_jaccard_threshold=0.3,
    embed_sim_threshold=0.75,
    dedup_threshold=0.9,
    min_cluster_size=2,
    entity_fn=_entity_fn,
)


def test_시간창_엔티티_모두_겹치면_한_클러스터로_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "NVIDIA 공급 이슈 후속", BASE_TIME + timedelta(hours=10), ["NVIDIA"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "NVIDIA 공급 이슈 후속": [0.99, 0.01]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    narrow = [c for c in clusters if c.window_type == "narrow"]
    assert len(narrow) == 1
    assert {a["id"] for a in narrow[0].articles} == {1, 2}
    assert unclustered == []


def test_시간창을_벗어나면_안_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "NVIDIA 공급 이슈 후속", BASE_TIME + timedelta(days=40), ["NVIDIA"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "NVIDIA 공급 이슈 후속": [0.99, 0.01]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 2


def test_엔티티가_다르면_안_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "삼성 반도체 투자", BASE_TIME + timedelta(hours=1), ["Samsung"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "삼성 반도체 투자": [0.0, 1.0]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 2


def test_min_cluster_size_미만이면_unclustered로_간다():
    articles = [_article(1, "단독 기사", BASE_TIME, ["NVIDIA"])]
    embed_fn = _make_embed_fn({"단독 기사": [1.0, 0.0]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 1


def test_근접중복_클러스터는_병합된다():
    # 엔티티가 달라 자카드로는 안 묶이지만, 임베딩이 거의 같은 두 그룹은 근접중복으로 병합된다.
    articles = [
        _article(1, "A사 공급 확대", BASE_TIME, ["A사"]),
        _article(2, "A사 공급 후속", BASE_TIME + timedelta(hours=1), ["A사"]),
        _article(3, "동일 사건 다른 표현", BASE_TIME + timedelta(hours=2), ["B사"]),
        _article(4, "동일 사건 다른 표현 후속", BASE_TIME + timedelta(hours=3), ["B사"]),
    ]
    embed_fn = _make_embed_fn(
        {
            "A사 공급 확대": [1.0, 0.0],
            "A사 공급 후속": [0.99, 0.01],
            "동일 사건 다른 표현": [0.98, 0.02],
            "동일 사건 다른 표현 후속": [0.97, 0.03],
        }
    )
    clusters, _ = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    narrow = [c for c in clusters if c.window_type == "narrow"]
    assert len(narrow) == 1
    assert {a["id"] for a in narrow[0].articles} == {1, 2, 3, 4}


def test_날짜_없는_기사는_unclustered에_포함된다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        {"id": 2, "title": "날짜 없는 기사", "description": "", "published_at": None, "entity": None, "event": None},
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert {a["id"] for a in unclustered} == {1, 2}
