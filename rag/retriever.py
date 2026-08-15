"""vector, text, weighted RRF hybrid 검색."""
import argparse
import json
from typing import Literal

from pgvector import Vector
from pgvector.psycopg import register_vector

from config import config
from db.db import get_conn

from .embed import embed_query

SearchMode = Literal["vector", "text", "hybrid"]
Metric = Literal["cosine", "l2", "inner_product"]
FusionMethod = Literal["rrf", "normalized"]

_VECTOR_METRICS = {
    "cosine": {"operator": "<=>", "score": "1 - distance"},
    "l2": {"operator": "<->", "score": "-distance"},
    "inner_product": {"operator": "<#>", "score": "-distance"},
}


def _validate_common(query: str, top_k: int) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query는 비어 있을 수 없습니다.")
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    return query


def _vector_search(query: str, limit: int, metric: Metric) -> list[dict]:
    if metric not in _VECTOR_METRICS:
        raise ValueError(f"지원하지 않는 vector metric: {metric}")
    query_vector = Vector(embed_query(query))
    operator = _VECTOR_METRICS[metric]["operator"]
    score_expression = _VECTOR_METRICS[metric]["score"]

    # SQL operator는 parameter binding이 불가능하므로 위 허용 목록에서만 선택한다.
    sql = f"""
        WITH candidates AS (
            SELECT
                ra.id AS article_id,
                ac.id AS chunk_id,
                ac.chunk_index,
                ac.chunk_text AS text,
                ra.title,
                ra.url,
                ce.embedding {operator} %(query_vector)s AS distance
            FROM chunk_embeddings AS ce
            JOIN article_chunks AS ac ON ac.id = ce.chunk_id
            JOIN raw_articles AS ra ON ra.id = ac.article_id
            WHERE ce.embedding_model = %(embedding_model)s
              AND ce.embedding_dimension = %(embedding_dimension)s
        )
        SELECT *, {score_expression} AS vector_score
        FROM candidates
        ORDER BY distance
        LIMIT %(limit)s
    """
    with get_conn() as conn:
        register_vector(conn)
        return conn.execute(
            sql,
            {
                "query_vector": query_vector,
                "embedding_model": config.HF_EMBEDDING_MODEL,
                "embedding_dimension": config.HF_EMBEDDING_DIMENSION,
                "limit": limit,
            },
        ).fetchall()


def _text_search(query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """WITH query AS (
                   SELECT websearch_to_tsquery('simple', %(query)s) AS value
               )
               SELECT
                   ra.id AS article_id,
                   ac.id AS chunk_id,
                   ac.chunk_index,
                   ac.chunk_text AS text,
                   ra.title,
                   ra.url,
                   ts_rank_cd(
                       to_tsvector('simple', ac.chunk_text),
                       query.value
                   ) AS text_score
               FROM article_chunks AS ac
               JOIN raw_articles AS ra ON ra.id = ac.article_id
               CROSS JOIN query
               WHERE to_tsvector('simple', ac.chunk_text) @@ query.value
               ORDER BY text_score DESC
               LIMIT %(limit)s""",
            {"query": query, "limit": limit},
        ).fetchall()


def _minmax(rows: list[dict], field: str) -> dict[int, float]:
    if not rows:
        return {}
    values = [float(row[field]) for row in rows]
    low, high = min(values), max(values)
    if high == low:
        return {row["chunk_id"]: 1.0 for row in rows}
    return {
        row["chunk_id"]: (float(row[field]) - low) / (high - low)
        for row in rows
    }


def _max_scale(rows: list[dict], field: str) -> dict[int, float]:
    """0이 자연스러운 최솟값인 양수 점수를 최댓값 기준으로 정규화한다."""
    if not rows:
        return {}
    high = max(float(row[field]) for row in rows)
    if high <= 0:
        return {row["chunk_id"]: 0.0 for row in rows}
    return {row["chunk_id"]: float(row[field]) / high for row in rows}


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
                for key in ("article_id", "chunk_id", "chunk_index", "text", "title", "url")
            }
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


def _combine_normalized_scores(
    vector_rows: list[dict],
    text_rows: list[dict],
    vector_weight: float,
    text_weight: float,
    top_k: int,
) -> list[dict]:
    vector_weight, text_weight = _normalized_weights(vector_weight, text_weight)

    vector_normalized = _minmax(vector_rows, "vector_score")
    # ts_rank_cd는 0 이상이므로 후보군의 최솟값 대신 실제 최솟값 0을 기준으로 둔다.
    text_normalized = _max_scale(text_rows, "text_score")
    merged = _merge_candidates(vector_rows, text_rows)

    for chunk_id, row in merged.items():
        row["vector_score_normalized"] = vector_normalized.get(chunk_id, 0.0)
        row["text_score_normalized"] = text_normalized.get(chunk_id, 0.0)
        row["score"] = (
            vector_weight * row["vector_score_normalized"]
            + text_weight * row["text_score_normalized"]
        )

    return sorted(merged.values(), key=lambda row: row["score"], reverse=True)[:top_k]


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


def retrieve(
    query: str,
    top_k: int = 10,
    search_mode: SearchMode = "vector",
    metric: Metric = "cosine",
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    fusion_method: FusionMethod = "rrf",
    candidate_k: int | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """문자열 query를 받아 vector, text 또는 hybrid 검색을 수행한다."""
    query = _validate_common(query, top_k)
    if search_mode not in {"vector", "text", "hybrid"}:
        raise ValueError(f"지원하지 않는 search_mode: {search_mode}")

    if search_mode == "vector":
        rows = _vector_search(query, top_k, metric)
        return [{**row, "score": float(row["vector_score"])} for row in rows]
    if search_mode == "text":
        rows = _text_search(query, top_k)
        return [{**row, "score": float(row["text_score"])} for row in rows]

    if fusion_method not in {"rrf", "normalized"}:
        raise ValueError(f"지원하지 않는 fusion_method: {fusion_method}")
    if candidate_k is not None and candidate_k < top_k:
        raise ValueError("candidate_k는 top_k 이상이어야 합니다.")
    if fusion_method == "rrf" and rrf_k <= 0:
        raise ValueError("rrf_k는 1 이상이어야 합니다.")
    _normalized_weights(vector_weight, text_weight)

    candidate_limit = candidate_k if candidate_k is not None else max(top_k * 10, 50)
    vector_rows = _vector_search(query, candidate_limit, metric)
    text_rows = _text_search(query, candidate_limit)
    if fusion_method == "normalized":
        return _combine_normalized_scores(
            vector_rows, text_rows, vector_weight, text_weight, top_k
        )
    return _combine_rrf_scores(
        vector_rows, text_rows, vector_weight, text_weight, top_k, rrf_k
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BrieFYI RAG 검색")
    parser.add_argument("--query", help="검색 문자열. 생략하면 터미널에서 입력")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("vector", "text", "hybrid"), default="vector")
    parser.add_argument(
        "--metric", choices=("cosine", "l2", "inner_product"), default="cosine"
    )
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--fusion", choices=("rrf", "normalized"), default="rrf")
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args(argv)

    query = args.query if args.query is not None else input("검색어를 입력하세요: ")
    rows = retrieve(
        query=query,
        top_k=args.top_k,
        search_mode=args.mode,
        metric=args.metric,
        vector_weight=args.vector_weight,
        text_weight=args.text_weight,
        fusion_method=args.fusion,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
