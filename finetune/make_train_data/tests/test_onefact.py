from make_train_data.clustering import Cluster
from make_train_data.onefact import select_onefact_candidates


def _article(id_, event=None):
    return {"id": id_, "title": f"기사{id_}", "event": event}


def test_크기1_클러스터와_unclustered가_후보풀이_된다():
    clusters = [Cluster(cluster_id="c1", window_type="narrow", articles=[_article(1)], entities=[], event_type=None)]
    unclustered = [_article(2), _article(3)]
    selected = select_onefact_candidates(clusters, unclustered, target_ratio=1.0, total_pool_size=10)
    assert {a["id"] for a in selected} == {1, 2, 3}


def test_정기_발표성_event가_우선순위를_받는다():
    clusters = []
    unclustered = [
        _article(1, event=["실적발표"]),
        _article(2, event=["규제"]),
        _article(3, event=["제품출시"]),
    ]
    selected = select_onefact_candidates(clusters, unclustered, target_ratio=1.0, total_pool_size=2)
    selected_ids = {a["id"] for a in selected}
    assert selected_ids == {1, 3}  # 정기 발표성(실적발표/제품출시)이 "규제"보다 우선


def test_target_ratio로_선정_개수를_제한한다():
    unclustered = [_article(i) for i in range(10)]
    selected = select_onefact_candidates([], unclustered, target_ratio=0.2, total_pool_size=10)
    assert len(selected) == 2  # round(10 * 0.2)


def test_후보가_없으면_빈_리스트():
    selected = select_onefact_candidates([], [], target_ratio=0.175, total_pool_size=10)
    assert selected == []
