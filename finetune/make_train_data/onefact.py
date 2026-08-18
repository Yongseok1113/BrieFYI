"""단발성 사실 기사 탐지·선정.

크기 1 클러스터 + 어디에도 클러스터링되지 못한 기사를 후보 풀로 삼고,
enrichment.event가 "정기 발표성"이면 우선순위를 높여 target_ratio만큼 선정한다.
enrichment.event는 data_pipeline이 LLM+synonym_table로 정규화한 값이라 고정
enum이 아니므로, ROUTINE_EVENT_TYPES는 코드 상수로 작게 시작해 synonym_table에
실제로 쌓인 canonical_value를 보고 조정하는 것을 전제로 한다. 최종 확정
(no_strong_insight 여부)은 이 함수가 하지 않는다 — 선정만 하고, 실제 라벨은
cluster_export.py가 만든 파일을 사람이 Claude 채팅에서 판단한다.
"""
from __future__ import annotations

ROUTINE_EVENT_TYPES = {"실적발표", "제품출시", "일반사실"}


def _is_routine(article: dict) -> bool:
    events = article.get("event") or []
    if isinstance(events, str):
        events = [events]
    return any(e in ROUTINE_EVENT_TYPES for e in events)


def select_onefact_candidates(
    clusters: list,
    unclustered_articles: list[dict],
    *,
    target_ratio: float,
    total_pool_size: int,
) -> list[dict]:
    pool: list[dict] = list(unclustered_articles)
    for cluster in clusters:
        if len(cluster.articles) == 1:
            pool.extend(cluster.articles)

    if not pool:
        return []

    routine = [a for a in pool if _is_routine(a)]
    rest = [a for a in pool if not _is_routine(a)]
    ordered = routine + rest  # 정기 발표성을 앞으로 정렬해 우선 선정되게 한다

    target_count = round(total_pool_size * target_ratio)
    target_count = max(0, min(target_count, len(ordered)))
    return ordered[:target_count]
