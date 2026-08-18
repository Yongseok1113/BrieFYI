import json
import tempfile
from pathlib import Path

from make_train_data.clustering import Cluster
from make_train_data.cluster_export import export_clusters


def _article(id_, title="제목", event_type=None):
    return {
        "id": id_, "title": title, "description": "설명", "url": f"https://x/{id_}",
        "source": "test", "published_at": None, "insights": None, "category": None, "domain": None,
    }


def test_클러스터당_파일_하나가_생성된다():
    clusters = [
        Cluster(cluster_id="c1", window_type="narrow", articles=[_article(1), _article(2)], entities=["NVIDIA"], event_type="공급망"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters(clusters, [], out_dir=Path(tmp))
        assert len(paths) == 1
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["cluster_id"] == "c1"
        assert data["window_type"] == "narrow"
        assert len(data["articles"]) == 2
        assert "claude_prompt" in data
        assert data["no_strong_insight_hint"] is False


def test_단발성_기사는_no_strong_insight_hint_true로_별도_파일에_들어간다():
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters([], [_article(1), _article(2)], out_dir=Path(tmp))
        assert len(paths) == 1
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["no_strong_insight_hint"] is True
        assert len(data["articles"]) == 2


def test_taxonomy_balance가_켜지면_한쪽으로_쏠린_event_type을_일부_제외한다():
    clusters = [
        Cluster(cluster_id=f"c{i}", window_type="narrow", articles=[_article(i), _article(i + 100)], entities=[], event_type="M&A")
        for i in range(10)
    ] + [
        Cluster(cluster_id="c_rare", window_type="narrow", articles=[_article(999), _article(998)], entities=[], event_type="규제")
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters(clusters, [], out_dir=Path(tmp), taxonomy_balance=True)
        event_types = [json.loads(p.read_text(encoding="utf-8"))["event_type"] for p in paths]
        assert "규제" in event_types  # 희소 유형은 반드시 포함
        assert event_types.count("M&A") < 10  # 압도적 다수인 유형은 일부 제외됨
