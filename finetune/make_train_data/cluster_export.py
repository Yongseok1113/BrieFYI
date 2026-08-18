"""클러스터(+단발성 기사)를 Claude 채팅에 붙여넣을 JSON 파일로 저장한다.

taxonomy_balance=True면 event_type 분포를 보고 특정 유형이 압도적으로 많을 때
초과분을 샘플링에서 제외한다(원 설계문서 5절 골드셋 규칙을 학습셋 생성에도 적용) —
가장 흔한 유형이라도 최소 개수는 남기고, 나머지 유형 개수의 평균을 넘는 만큼만 잘라낸다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .clustering import Cluster

CLAUDE_PROMPT_TEMPLATE = (
    "다음 기사 묶음이 정말 같은 사건/테마를 다루는지 먼저 확인해줘. 다루지 않는다면 "
    "어떻게 나눠야 하는지 알려줘. 같은 사건/테마가 맞다면, facts/insights"
    "(전략적_의도|모순_긴장|선행지표|이해관계자_영향|시장_신호|리스크_기회 중 최소 2종 이상 "
    "시도, 억지로 6종 다 채우지 않음)/no_strong_insight를 판단해서 아래 스키마로 답해줘.\n\n"
    '{"facts": ["..."], "insights": [{"claim": "...", "type": "...", "evidence": ["..."], '
    '"confidence": "high|medium|low", "reasoning": "..."}], "no_strong_insight": false}\n\n'
    "추가로 이 클러스터에 대해 종합적 판단을 요구하는 테스트용 질문과 모범 답안도 하나 만들어줘."
)


def _article_payload(article: dict) -> dict:
    payload = {
        "title": article.get("title"),
        "description": article.get("description"),
        "source": article.get("source"),
        "published_at": str(article.get("published_at")) if article.get("published_at") else None,
        "url": article.get("url"),
    }
    if article.get("insights") or article.get("category") or article.get("domain"):
        payload["enrichment_hint"] = {
            "insights": article.get("insights"),
            "category": article.get("category"),
            "domain": article.get("domain"),
        }
    return payload


def _cluster_to_dict(cluster: Cluster, *, no_strong_insight_hint: bool) -> dict:
    return {
        "cluster_id": cluster.cluster_id,
        "window_type": cluster.window_type,
        "no_strong_insight_hint": no_strong_insight_hint,
        "entities": cluster.entities,
        "event_type": cluster.event_type,
        "articles": [_article_payload(a) for a in cluster.articles],
        "claude_prompt": CLAUDE_PROMPT_TEMPLATE,
    }


def _apply_taxonomy_balance(clusters: list[Cluster]) -> list[Cluster]:
    counts = Counter(c.event_type for c in clusters if c.event_type)
    if len(counts) <= 1:
        return clusters

    other_counts = sorted(v for k, v in counts.items())[:-1]
    cap = max(1, round(sum(other_counts) / len(other_counts))) if other_counts else max(counts.values())

    kept: list[Cluster] = []
    seen_count: dict[str, int] = {}
    for cluster in clusters:
        key = cluster.event_type
        if key is None:
            kept.append(cluster)
            continue
        seen_count[key] = seen_count.get(key, 0) + 1
        if seen_count[key] <= max(cap, 1):
            kept.append(cluster)
    return kept


def export_clusters(
    clusters: list[Cluster],
    onefact_articles: list[dict],
    *,
    out_dir: Path,
    taxonomy_balance: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clusters = _apply_taxonomy_balance(clusters) if taxonomy_balance else clusters

    paths: list[Path] = []
    for cluster in target_clusters:
        path = out_dir / f"{cluster.cluster_id}.json"
        path.write_text(
            json.dumps(_cluster_to_dict(cluster, no_strong_insight_hint=False), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(path)

    if onefact_articles:
        onefact_cluster = Cluster(
            cluster_id="c_onefact_batch",
            window_type="narrow",
            articles=onefact_articles,
            entities=[],
            event_type=None,
        )
        path = out_dir / "c_onefact_batch.json"
        path.write_text(
            json.dumps(_cluster_to_dict(onefact_cluster, no_strong_insight_hint=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(path)

    return paths
