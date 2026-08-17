from data_pipeline.clustering import cluster_values

_VECTORS = {
    "삼성전자": [1.0, 0.0, 0.0],
    "삼성": [0.95, 0.05, 0.0],
    "Samsung Electronics": [0.9, 0.1, 0.0],
    "NVIDIA": [0.0, 1.0, 0.0],
    "엔비디아": [0.0, 0.95, 0.05],
}


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [_VECTORS[t] for t in texts]


def test_similar_values_cluster_together():
    values_with_counts = [
        ("삼성전자", 10), ("NVIDIA", 8), ("삼성", 5), ("엔비디아", 3), ("Samsung Electronics", 2),
    ]
    clusters = cluster_values(values_with_counts, threshold=0.9, embed_fn=fake_embed)

    assert len(clusters) == 2
    by_canonical = {c.canonical: set(c.aliases) for c in clusters}
    assert by_canonical["삼성전자"] == {"삼성", "Samsung Electronics"}
    assert by_canonical["NVIDIA"] == {"엔비디아"}


def test_most_frequent_value_becomes_canonical():
    # "삼성"이 가장 빈도가 높으면 canonical이 돼야 한다 (등장 빈도 내림차순 처리).
    values_with_counts = [("삼성", 20), ("삼성전자", 5)]
    clusters = cluster_values(values_with_counts, threshold=0.9, embed_fn=fake_embed)

    assert len(clusters) == 1
    assert clusters[0].canonical == "삼성"
    assert clusters[0].aliases == ["삼성전자"]


def test_dissimilar_values_stay_separate():
    values_with_counts = [("삼성전자", 10), ("NVIDIA", 8)]
    clusters = cluster_values(values_with_counts, threshold=0.9, embed_fn=fake_embed)
    assert len(clusters) == 2


def test_empty_input_returns_empty_list():
    assert cluster_values([], threshold=0.9, embed_fn=fake_embed) == []


def test_alias_values_includes_canonical():
    values_with_counts = [("삼성전자", 10), ("삼성", 5)]
    clusters = cluster_values(values_with_counts, threshold=0.9, embed_fn=fake_embed)
    assert set(clusters[0].alias_values()) == {"삼성전자", "삼성"}
