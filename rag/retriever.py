"""vector, text, weighted RRF hybrid 검색.

점수 결합은 원점수 척도 대신 검색기별 순위를 쓰는 weighted RRF 하나로 고정한다. SQL은
`db.py`에 있고 이 모듈은 결합과 metadata 가산점 계산만 담당한다.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from . import db
from .embed import embed_query

SearchMode = Literal["vector", "text", "hybrid"]


def _validate_common(query: str, top_k: int) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query는 비어 있을 수 없습니다.")
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    return query


def _normalized_weights(vector_weight: float, text_weight: float) -> tuple[float, float]:
    if vector_weight < 0 or text_weight < 0 or vector_weight + text_weight <= 0:
        raise ValueError("검색 가중치는 0 이상이며 합이 0보다 커야 합니다.")
    weight_sum = vector_weight + text_weight
    return vector_weight / weight_sum, text_weight / weight_sum


def _merge_candidates(vector_rows: list[dict], text_rows: list[dict]) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for row in vector_rows + text_rows:
        chunk_id = row["chunk_id"]
        if chunk_id not in merged:
            merged[chunk_id] = {
                key: row[key]
                for key in (
                    "article_id",
                    "chunk_id",
                    "chunk_index",
                    "text",
                    "title",
                    "url",
                )
            }
            merged[chunk_id].update(
                {
                    "category": row.get("category"),
                    "domains": row.get("domains") or [],
                    "entities": row.get("entities") or [],
                }
            )
        if row.get("vector_score") is not None:
            merged[chunk_id]["vector_score"] = float(row["vector_score"])
        if row.get("text_score") is not None:
            merged[chunk_id]["text_score"] = float(row["text_score"])
    return merged


def _dense_ranks(rows: list[dict], field: str) -> dict[int, int]:
    """같은 원점수에는 같은 순위를 부여한다: 0.3, 0.3, 0.2 -> 1, 1, 2."""
    unique_scores = sorted({float(row[field]) for row in rows}, reverse=True)
    score_to_rank = {score: rank for rank, score in enumerate(unique_scores, start=1)}
    return {
        row["chunk_id"]: score_to_rank[float(row[field])]
        for row in rows
    }


def _combine_rrf_scores(
    vector_rows: list[dict],
    text_rows: list[dict],
    vector_weight: float,
    text_weight: float,
    top_k: int,
    rrf_k: int,
) -> list[dict]:
    """원점수 척도 대신 검색기별 dense rank를 weighted RRF로 결합한다."""
    if rrf_k <= 0:
        raise ValueError("rrf_k는 1 이상이어야 합니다.")
    vector_weight, text_weight = _normalized_weights(vector_weight, text_weight)
    vector_ranks = _dense_ranks(vector_rows, "vector_score")
    text_ranks = _dense_ranks(text_rows, "text_score")
    merged = _merge_candidates(vector_rows, text_rows)

    for chunk_id, row in merged.items():
        vector_rank = vector_ranks.get(chunk_id)
        text_rank = text_ranks.get(chunk_id)
        row["vector_rank"] = vector_rank
        row["text_rank"] = text_rank
        row["vector_rrf_score"] = (
            vector_weight / (rrf_k + vector_rank) if vector_rank is not None else 0.0
        )
        row["text_rrf_score"] = (
            text_weight / (rrf_k + text_rank) if text_rank is not None else 0.0
        )
        row["score"] = row["vector_rrf_score"] + row["text_rrf_score"]

    return sorted(
        merged.values(),
        key=lambda row: (-row["score"], row["chunk_id"]),
    )[:top_k]


def _normalize_metadata_query(
    category: str | None,
    domains: Iterable[str] | None,
) -> tuple[str | None, list[str]]:
    normalized_category = category.strip() if category and category.strip() else None
    normalized_domains = list(
        dict.fromkeys(domain.strip() for domain in (domains or []) if domain.strip())
    )
    return normalized_category, normalized_domains


def _apply_metadata_boost(
    rows: list[dict],
    *,
    category: str | None,
    domains: list[str],
    category_boost: float,
    domain_boost: float,
    top_k: int,
) -> list[dict]:
    """후보를 제거하지 않고 Category/Domain 일치에 score-scale 비례 가산점을 준다."""
    if category_boost < 0 or domain_boost < 0:
        raise ValueError("metadata boost는 0 이상이어야 합니다.")
    if not rows:
        return []

    scores = [float(row["score"]) for row in rows]
    score_scale = max(max(scores) - min(scores), abs(max(scores)), 1e-12)
    boosted: list[dict] = []
    for row in rows:
        article_domains = list(row.get("domains") or [])
        category_match = category is not None and row.get("category") == category
        matched_domains = [domain for domain in domains if domain in article_domains]
        domain_match_fraction = len(matched_domains) / len(domains) if domains else 0.0
        metadata_score = score_scale * (
            category_boost * int(category_match)
            + domain_boost * domain_match_fraction
        )
        boosted.append(
            {
                **row,
                "base_score": float(row["score"]),
                "metadata_score": metadata_score,
                "category_match": category_match,
                "matched_domains": matched_domains,
                "score": float(row["score"]) + metadata_score,
            }
        )
    return sorted(
        boosted,
        key=lambda row: (-row["score"], row["chunk_id"]),
    )[:top_k]


def retrieve(
    query: str,
    top_k: int = 10,
    search_mode: SearchMode = "vector",
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    category: str | None = None,
    domains: Iterable[str] | None = None,
    category_boost: float = 0.05,
    domain_boost: float = 0.05,
) -> list[dict]:
    """검색 후 Category/Domain 일치 후보에 가산점을 적용한다."""
    query = _validate_common(query, top_k)
    category, normalized_domains = _normalize_metadata_query(category, domains)
    if category_boost < 0 or domain_boost < 0:
        raise ValueError("metadata boost는 0 이상이어야 합니다.")
    use_metadata_boost = category is not None or bool(normalized_domains)
    if search_mode not in {"vector", "text", "hybrid"}:
        raise ValueError(f"지원하지 않는 search_mode: {search_mode}")

    if search_mode in {"vector", "text"}:
        limit = max(top_k * 10, 50) if use_metadata_boost else top_k
        if search_mode == "vector":
            rows = db.vector_search(embed_query(query), limit)
            scored = [{**row, "score": float(row["vector_score"])} for row in rows]
        else:
            rows = db.text_search(query, limit)
            scored = [{**row, "score": float(row["text_score"])} for row in rows]
        return _apply_metadata_boost(
            scored,
            category=category,
            domains=normalized_domains,
            category_boost=category_boost,
            domain_boost=domain_boost,
            top_k=top_k,
        )

    if candidate_k is not None and candidate_k < top_k:
        raise ValueError("candidate_k는 top_k 이상이어야 합니다.")
    if rrf_k <= 0:
        raise ValueError("rrf_k는 1 이상이어야 합니다.")
    _normalized_weights(vector_weight, text_weight)

    candidate_limit = candidate_k if candidate_k is not None else max(top_k * 10, 50)
    vector_rows = db.vector_search(embed_query(query), candidate_limit)
    text_rows = db.text_search(query, candidate_limit)
    merge_limit = candidate_limit if use_metadata_boost else top_k
    scored = _combine_rrf_scores(
        vector_rows, text_rows, vector_weight, text_weight, merge_limit, rrf_k
    )
    return _apply_metadata_boost(
        scored,
        category=category,
        domains=normalized_domains,
        category_boost=category_boost,
        domain_boost=domain_boost,
        top_k=top_k,
    )
