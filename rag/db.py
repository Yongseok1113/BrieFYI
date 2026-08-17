"""RAG 전용 SQL을 한곳에 모은 모듈.

develop의 `data_pipeline/db.py`와 같은 역할이지만, 연결 자체는 레포 루트 `db/db.py`의
`get_conn()`을 그대로 쓴다 — data_pipeline과 달리 rag는 메인 앱과 같은 이미지에서 같은
DB에 붙기 때문이다. 스키마는 `db/vector_schema.sql`에 정의돼 있고 여기서는 그 테이블에
접근만 한다.
"""
from __future__ import annotations

from collections.abc import Iterable

from pgvector import Vector
from pgvector.psycopg import register_vector

from config import config
from db.db import get_conn

from .taxonomy import (
    EVENT_EXTRACTION_VERSION,
    EVENT_TAXONOMY_VERSION,
    GLINER2_MODEL_NAME,
)


def normalize_article_ids(article_ids: Iterable[int]) -> list[int]:
    """호출자가 넘긴 ID를 순서를 지키며 중복 없는 정수 목록으로 만든다."""
    return list(dict.fromkeys(int(article_id) for article_id in article_ids))


def _require_all_found(requested_ids: list[int], found_ids: set[int]) -> None:
    missing_ids = [article_id for article_id in requested_ids if article_id not in found_ids]
    if missing_ids:
        raise ValueError(f"존재하지 않는 article_id: {missing_ids}")


# ---------------------------------------------------------------------------
# 기사 조회
# ---------------------------------------------------------------------------

def load_articles(article_ids: list[int]) -> list[dict]:
    """indexing 대상 기사를 조회한다. 없는 ID가 하나라도 있으면 거부한다."""
    with get_conn() as conn:
        articles = conn.execute(
            """SELECT id, title, description, url
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()
    _require_all_found(article_ids, {article["id"] for article in articles})
    return articles


def load_article_ids_by_urls(urls: list[str]) -> list[int]:
    if not urls:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id
               FROM raw_articles
               WHERE url = ANY(%s)
               ORDER BY id""",
            (urls,),
        ).fetchall()
    return [row["id"] for row in rows]


def load_unindexed_article_ids(limit: int) -> list[int]:
    """embedding이 아직 없는 기사만 제한된 수만큼 고른다."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT ra.id
               FROM raw_articles AS ra
               WHERE NOT EXISTS (
                   SELECT 1
                   FROM article_chunks AS ac
                   JOIN chunk_embeddings AS ce ON ce.chunk_id = ac.id
                   WHERE ac.article_id = ra.id
               )
               ORDER BY ra.id
               LIMIT %s""",
            (limit,),
        ).fetchall()
    return [row["id"] for row in rows]


def load_articles_with_chunks(article_ids: list[int]) -> list[dict]:
    """Event indexing 대상 기사를 저장된 chunk와 함께 조회한다."""
    with get_conn() as conn:
        article_rows = conn.execute(
            """SELECT id, title
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()
        chunk_rows = conn.execute(
            """SELECT id, article_id, chunk_index, chunk_text
               FROM article_chunks
               WHERE article_id = ANY(%s)
               ORDER BY article_id, chunk_index""",
            (article_ids,),
        ).fetchall()
    _require_all_found(article_ids, {article["id"] for article in article_rows})

    chunks_by_article: dict[int, list[dict]] = {}
    for chunk in chunk_rows:
        chunks_by_article.setdefault(chunk["article_id"], []).append(chunk)
    return [
        {
            "article_id": article["id"],
            "title": article["title"],
            "chunks": chunks_by_article.get(article["id"], []),
        }
        for article in article_rows
    ]


# ---------------------------------------------------------------------------
# chunk / embedding / 4-Layer metadata 저장
# ---------------------------------------------------------------------------

def save_article_topics(
    article_id: int,
    category: str,
    domains: list[str],
    entities: list[str],
    *,
    conn=None,
) -> None:
    """기사의 Category/Domain/Entity만 article_id 기준으로 UPSERT한다.

    `topic_text`와 topic `embedding`은 후속 실험용 컬럼이라 UPDATE 대상에서 제외해,
    나중에 값이 채워져도 metadata 재저장이 그 값을 지우지 않게 한다.
    """
    def execute(target_conn):
        target_conn.execute(
            """INSERT INTO article_topics
                   (article_id, category, domains, entities)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (article_id) DO UPDATE SET
                   category = EXCLUDED.category,
                   domains = EXCLUDED.domains,
                   entities = EXCLUDED.entities""",
            (article_id, category, domains, entities),
        )

    if conn is not None:
        execute(conn)
        return
    with get_conn() as managed_conn:
        execute(managed_conn)


def store_article_index(
    article_id: int,
    chunks: list[dict],
    vectors: list[list[float]],
    topics: dict,
) -> None:
    """한 기사의 chunk/embedding/metadata를 같은 transaction에서 교체한다."""
    with get_conn() as conn:
        register_vector(conn)
        existing_rows = conn.execute(
            """SELECT id, chunk_index, chunk_text
               FROM article_chunks
               WHERE article_id = %s
               ORDER BY chunk_index""",
            (article_id,),
        ).fetchall()
        existing_by_index = {row["chunk_index"]: row for row in existing_rows}
        existing_signature = [
            (row["chunk_index"], row["chunk_text"]) for row in existing_rows
        ]
        new_signature = [
            (chunk["chunk_index"], chunk["chunk_text"]) for chunk in chunks
        ]

        if existing_signature != new_signature:
            # Event와 argument는 이전 chunk text에서 파생된 값이므로 함께 무효화한다.
            conn.execute(
                "DELETE FROM article_event_index_status WHERE article_id = %s",
                (article_id,),
            )
            conn.execute(
                "DELETE FROM article_events WHERE article_id = %s",
                (article_id,),
            )

        current_indexes: list[int] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            existing = existing_by_index.get(chunk["chunk_index"])
            if existing is not None and existing["chunk_text"] != chunk["chunk_text"]:
                # 같은 chunk_id의 다른 model embedding도 더 이상 유효하지 않다.
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE chunk_id = %s",
                    (existing["id"],),
                )

            chunk_id = conn.execute(
                """INSERT INTO article_chunks (article_id, chunk_index, chunk_text)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (article_id, chunk_index) DO UPDATE SET
                       chunk_text = EXCLUDED.chunk_text
                   RETURNING id""",
                (article_id, chunk["chunk_index"], chunk["chunk_text"]),
            ).fetchone()["id"]
            current_indexes.append(chunk["chunk_index"])

            conn.execute(
                """INSERT INTO chunk_embeddings
                       (chunk_id, embedding_model, embedding_dimension, embedding)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (chunk_id, embedding_model) DO UPDATE SET
                       embedding_dimension = EXCLUDED.embedding_dimension,
                       embedding = EXCLUDED.embedding,
                       created_at = now()""",
                (
                    chunk_id,
                    config.HF_EMBEDDING_MODEL,
                    config.HF_EMBEDDING_DIMENSION,
                    Vector(vector),
                ),
            )

        conn.execute(
            """DELETE FROM article_chunks
               WHERE article_id = %s AND NOT (chunk_index = ANY(%s))""",
            (article_id, current_indexes),
        )
        save_article_topics(
            article_id,
            topics["category"],
            topics["domains"],
            topics["entities"],
            conn=conn,
        )


# ---------------------------------------------------------------------------
# 구조화 Event 저장과 상태
# ---------------------------------------------------------------------------

def load_event_statuses(article_ids: list[int]) -> dict[int, dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT article_id, extractor_model, taxonomy_version,
                      extraction_version, source_fingerprint, status, event_count
               FROM article_event_index_status
               WHERE article_id = ANY(%s)""",
            (article_ids,),
        ).fetchall()
    return {row["article_id"]: row for row in rows}


def replace_article_events(
    article_id: int,
    source_fingerprint: str,
    events: list[dict],
) -> None:
    """한 기사 결과 전체와 completed 상태를 같은 transaction에서 교체한다."""
    with get_conn() as conn:
        conn.execute("DELETE FROM article_events WHERE article_id = %s", (article_id,))

        for event in events:
            event_id = conn.execute(
                """INSERT INTO article_events
                       (article_id, source_chunk_id, event_type, confidence,
                        extractor_model, taxonomy_version, extraction_version,
                        event_fingerprint, evidence_start, evidence_end)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    article_id,
                    event["source_chunk_id"],
                    event["event_type"],
                    event["confidence"],
                    GLINER2_MODEL_NAME,
                    EVENT_TAXONOMY_VERSION,
                    EVENT_EXTRACTION_VERSION,
                    event["event_fingerprint"],
                    event["evidence_start"],
                    event["evidence_end"],
                ),
            ).fetchone()["id"]

            for argument_index, argument in enumerate(event["arguments"]):
                conn.execute(
                    """INSERT INTO article_event_arguments
                           (event_id, argument_index, role, text, normalized_text,
                            confidence, span_start, span_end)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        event_id,
                        argument_index,
                        argument["role"],
                        argument["text"],
                        argument["normalized_text"],
                        argument["confidence"],
                        argument["span_start"],
                        argument["span_end"],
                    ),
                )

        conn.execute(
            """INSERT INTO article_event_index_status
                   (article_id, extractor_model, taxonomy_version, extraction_version,
                    source_fingerprint, status, event_count, error, processed_at)
               VALUES (%s, %s, %s, %s, %s, 'completed', %s, NULL, now())
               ON CONFLICT (article_id) DO UPDATE SET
                   extractor_model = EXCLUDED.extractor_model,
                   taxonomy_version = EXCLUDED.taxonomy_version,
                   extraction_version = EXCLUDED.extraction_version,
                   source_fingerprint = EXCLUDED.source_fingerprint,
                   status = EXCLUDED.status,
                   event_count = EXCLUDED.event_count,
                   error = NULL,
                   processed_at = now()""",
            (
                article_id,
                GLINER2_MODEL_NAME,
                EVENT_TAXONOMY_VERSION,
                EVENT_EXTRACTION_VERSION,
                source_fingerprint,
                len(events),
            ),
        )


def mark_event_failed(article_id: int, source_fingerprint: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO article_event_index_status
                   (article_id, extractor_model, taxonomy_version, extraction_version,
                    source_fingerprint, status, event_count, error, processed_at)
               VALUES (%s, %s, %s, %s, %s, 'failed', 0, %s, now())
               ON CONFLICT (article_id) DO UPDATE SET
                   extractor_model = EXCLUDED.extractor_model,
                   taxonomy_version = EXCLUDED.taxonomy_version,
                   extraction_version = EXCLUDED.extraction_version,
                   source_fingerprint = EXCLUDED.source_fingerprint,
                   status = EXCLUDED.status,
                   event_count = 0,
                   error = EXCLUDED.error,
                   processed_at = now()""",
            (
                article_id,
                GLINER2_MODEL_NAME,
                EVENT_TAXONOMY_VERSION,
                EVENT_EXTRACTION_VERSION,
                source_fingerprint,
                error,
            ),
        )


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def vector_search(query_vector: list[float], limit: int) -> list[dict]:
    """저장된 chunk embedding을 cosine 거리로 검색한다.

    문서 vector와 query vector는 embed.py에서 모두 L2 정규화되므로, cosine 외의
    거리 연산자를 써도 순위가 같다. 그래서 연산자는 `<=>` 하나로 고정한다.
    """
    with get_conn() as conn:
        register_vector(conn)
        return conn.execute(
            """WITH candidates AS (
                   SELECT
                       ra.id AS article_id,
                       ac.id AS chunk_id,
                       ac.chunk_index,
                       ac.chunk_text AS text,
                       ra.title,
                       ra.url,
                       at.category,
                       at.domains,
                       at.entities,
                       ce.embedding <=> %(query_vector)s AS distance
                   FROM chunk_embeddings AS ce
                   JOIN article_chunks AS ac ON ac.id = ce.chunk_id
                   JOIN raw_articles AS ra ON ra.id = ac.article_id
                   LEFT JOIN article_topics AS at ON at.article_id = ra.id
                   WHERE ce.embedding_model = %(embedding_model)s
                     AND ce.embedding_dimension = %(embedding_dimension)s
               )
               SELECT *, 1 - distance AS vector_score
               FROM candidates
               ORDER BY distance
               LIMIT %(limit)s""",
            {
                "query_vector": Vector(query_vector),
                "embedding_model": config.HF_EMBEDDING_MODEL,
                "embedding_dimension": config.HF_EMBEDDING_DIMENSION,
                "limit": limit,
            },
        ).fetchall()


def text_search(query: str, limit: int) -> list[dict]:
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
                   at.category,
                   at.domains,
                   at.entities,
                   ts_rank_cd(
                       to_tsvector('simple', ac.chunk_text),
                       query.value
                   ) AS text_score
               FROM article_chunks AS ac
               JOIN raw_articles AS ra ON ra.id = ac.article_id
               LEFT JOIN article_topics AS at ON at.article_id = ra.id
               CROSS JOIN query
               WHERE to_tsvector('simple', ac.chunk_text) @@ query.value
               ORDER BY text_score DESC
               LIMIT %(limit)s""",
            {"query": query, "limit": limit},
        ).fetchall()
