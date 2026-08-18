"""기사 연관관계 클러스터링 (원 설계문서 3절 1~3단계 + 근접중복 병합).

시간창 필터 -> 엔티티 자카드 후보군(union-find) -> 임베딩 정제(그리디 threshold,
data_pipeline/clustering.py와 동일 전략) -> 근접중복 병합 순으로 실행한다.
원 설계문서 3절의 4단계("LLM 검증/병합", Claude 자동 호출)는 이 파이프라인에서
자동화하지 않는다 — 근접중복 병합으로 대체하고, 실제 검증은 cluster_export.py가
만드는 파일을 사람이 Claude 채팅에 전달하는 수동 단계에서 이뤄진다.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]
EntityFn = Callable[[dict], tuple[list[str], str | None, list[str]]]


@dataclass
class Cluster:
    cluster_id: str
    window_type: str  # "narrow" | "broad"
    articles: list[dict]
    entities: list[str]
    event_type: str | None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _article_text(article: dict) -> str:
    title = article.get("title") or ""
    description = article.get("description") or ""
    text = f"{title} {description}".strip()
    return text or title or "(제목 없음)"


def _candidate_groups(
    articles: list[dict],
    entity_sets: list[set[str]],
    window: timedelta,
    jaccard_threshold: float,
) -> list[list[int]]:
    n = len(articles)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        pub_i = articles[i].get("published_at")
        for j in range(i + 1, n):
            pub_j = articles[j].get("published_at")
            if abs(pub_i - pub_j) > window:
                continue
            if _jaccard(entity_sets[i], entity_sets[j]) >= jaccard_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _refine_by_embedding(indices: list[int], embeddings: list[list[float]], threshold: float) -> list[list[int]]:
    if len(indices) <= 1:
        return [indices]

    sub_clusters: list[dict] = []
    for idx in indices:
        vec = embeddings[idx]
        best, best_score = None, -1.0
        for sc in sub_clusters:
            score = _cosine(vec, sc["centroid"])
            if score > best_score:
                best, best_score = sc, score
        if best is not None and best_score >= threshold:
            best["members"].append(idx)
        else:
            # "centroid"라 부르지만 첫 멤버의 원본 벡터를 그대로 쓰고 갱신하지 않는다
            # (data_pipeline/clustering.py의 그리디 canonical 패턴과 동일). 진짜 평균
            # centroid를 재계산하는 _merge_near_duplicates()의 같은 이름과 혼동하지 말 것.
            sub_clusters.append({"members": [idx], "centroid": vec})
    return [sc["members"] for sc in sub_clusters]


def _centroid(embeddings: list[list[float]], indices: list[int]) -> list[float]:
    vectors = [embeddings[i] for i in indices]
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]


def _groups_within_window(dated: list[dict], group_a: list[int], group_b: list[int], window: timedelta) -> bool:
    """두 그룹 중 어느 한 쌍이라도 시간창 안에 있으면 True. 근접중복 병합이 시간창을
    완전히 무시하고 임베딩만으로 멀리 떨어진 사건을 합치는 것을 막는다 — 이게 없으면
    1단계 시간창 필터가 4단계에서 무력화된다."""
    times_a = [dated[i]["published_at"] for i in group_a]
    times_b = [dated[i]["published_at"] for i in group_b]
    return any(abs(ta - tb) <= window for ta in times_a for tb in times_b)


def _merge_near_duplicates(
    groups: list[list[int]],
    embeddings: list[list[float]],
    dated: list[dict],
    window: timedelta,
    threshold: float,
) -> list[list[int]]:
    merged: list[dict] = []
    for group in groups:
        centroid = _centroid(embeddings, group)
        target = None
        for m in merged:
            if _groups_within_window(dated, group, m["members"], window) and _cosine(centroid, m["centroid"]) >= threshold:
                target = m
                break
        if target is not None:
            target["members"].extend(group)
            target["centroid"] = _centroid(embeddings, target["members"])
        else:
            merged.append({"members": list(group), "centroid": centroid})
    return [m["members"] for m in merged]


def cluster_articles(
    articles: list[dict],
    *,
    narrow_window_hours: float,
    broad_window_days: float,
    entity_jaccard_threshold: float,
    embed_sim_threshold: float,
    dedup_threshold: float,
    min_cluster_size: int,
    entity_fn: EntityFn,
    embed_fn: EmbedFn,
) -> tuple[list[Cluster], list[dict]]:
    """(클러스터 목록, 클러스터에 들지 못한 기사 목록)을 반환한다.

    narrow/broad 두 시간창은 서로 다른 목적(같은 사건 vs 같은 테마 흐름)이라 병행
    운영하며, 같은 기사가 양쪽에 각각 다른 클러스터로 들어갈 수 있다. "클러스터에
    들지 못했다"는 판정은 narrow 기준으로만 한다 — broad에만 걸리는 기사는 여전히
    개별 사건으로서는 고립돼 있어 단발성 후보로 취급하는 게 맞기 때문이다.
    """
    dated = [a for a in articles if a.get("published_at") is not None]
    undated = [a for a in articles if a.get("published_at") is None]
    if not dated:
        return [], undated

    entity_sets = [set(entity_fn(a)[0]) for a in dated]
    texts = [_article_text(a) for a in dated]
    embeddings = embed_fn(texts)

    all_clusters: list[Cluster] = []
    clustered_indices: set[int] = set()

    for window_type, window in (
        ("narrow", timedelta(hours=narrow_window_hours)),
        ("broad", timedelta(days=broad_window_days)),
    ):
        candidate_groups = _candidate_groups(dated, entity_sets, window, entity_jaccard_threshold)
        refined: list[list[int]] = []
        for group in candidate_groups:
            refined.extend(_refine_by_embedding(group, embeddings, embed_sim_threshold))
        final_groups = _merge_near_duplicates(refined, embeddings, dated, window, dedup_threshold)

        for i, group in enumerate(final_groups):
            if len(group) < min_cluster_size:
                continue
            member_articles = [dated[idx] for idx in group]
            union_entities = sorted(set().union(*(entity_sets[idx] for idx in group)))
            raw_events = [dated[idx].get("event") for idx in group if dated[idx].get("event")]
            flat_events = [e for sub in raw_events for e in (sub if isinstance(sub, list) else [sub])]
            event_type = Counter(flat_events).most_common(1)[0][0] if flat_events else None

            all_clusters.append(
                Cluster(
                    cluster_id=f"c_{window_type}_{i}",
                    window_type=window_type,
                    articles=member_articles,
                    entities=union_entities,
                    event_type=event_type,
                )
            )
            if window_type == "narrow":
                clustered_indices.update(group)

    unclustered = [dated[i] for i in range(len(dated)) if i not in clustered_indices] + undated
    return all_clusters, unclustered
