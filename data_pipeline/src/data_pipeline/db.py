"""DB 접근. 레포 루트 db/db.py와 로직은 비슷하지만 완전히 독립된 모듈이다 —
data_pipeline은 별도 Docker 이미지라 레포 루트 코드를 import할 수 없다(8절).
스키마 자체(raw_articles 확장 컬럼, enrichment, synonym_table)는 db/schema.sql
(레포 루트, 메인 앱과 공유)에 정의돼 있고 여기서는 그 테이블에 접근만 한다.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import config

# raw_domain/raw_entity/raw_event는 JSONB 배열, raw_category는 TEXT
_ARRAY_DIMENSIONS = {"domain": "raw_domain", "entity": "raw_entity", "event": "raw_event"}
_SCALAR_DIMENSIONS = {"category": "raw_category"}


@contextmanager
def get_conn():
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 변형1: 수집(ingest) + 키워드(extract)
# ---------------------------------------------------------------------------

def insert_raw_article(digest_date: str, fields: dict, *, status: str = "pending") -> int | None:
    """구조화/비구조화 소스 공통 삽입 지점. 중복 url은 무시. 신규 삽입 시 id, 중복이면 None."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO raw_articles
               (digest_date, title, description, url, source, published_at, pipeline_status)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (url) DO NOTHING
               RETURNING id""",
            (
                digest_date,
                fields.get("title", ""),
                fields.get("description", ""),
                fields.get("url", ""),
                fields.get("source", ""),
                fields.get("published_at") or None,
                status,
            ),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def fetch_articles_by_status(status: str, limit: int) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, description, url, source, published_at FROM raw_articles "
            "WHERE pipeline_status = %s ORDER BY id LIMIT %s",
            (status, limit),
        )
        return cur.fetchall()


def set_article_status(article_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE raw_articles SET pipeline_status = %s WHERE id = %s", (status, article_id)
        )


def set_article_keywords(article_id: int, keywords: list[str]) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE raw_articles SET keywords = %s, pipeline_status = 'extracted' WHERE id = %s",
            (Jsonb(keywords), article_id),
        )


def mark_failed(article_id: int, stage: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE raw_articles SET pipeline_status = 'failed' WHERE id = %s", (article_id,)
        )
    print(f"[data_pipeline] article {article_id} failed at stage={stage}: {error}")


# ---------------------------------------------------------------------------
# 변형2: enrich (원시값 저장)
# ---------------------------------------------------------------------------

def insert_enrichment_raw(
    raw_article_id: int,
    insights: list[dict],
    implications: list[str],
    raw_category: str | None,
    raw_domain: list[str],
    raw_entity: list[str],
    raw_event: list[str],
    model_used: str,
    prompt_version: str,
) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO enrichment
               (raw_article_id, insights, implications, raw_category, raw_domain, raw_entity, raw_event,
                model_used, prompt_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                raw_article_id,
                Jsonb(insights),
                Jsonb(implications),
                raw_category,
                Jsonb(raw_domain),
                Jsonb(raw_entity),
                Jsonb(raw_event),
                model_used,
                prompt_version,
            ),
        )
        enrichment_id = cur.fetchone()["id"]
        cur.execute(
            "UPDATE raw_articles SET pipeline_status = 'enriched' WHERE id = %s", (raw_article_id,)
        )
        return enrichment_id


# ---------------------------------------------------------------------------
# 변형3: normalize
# ---------------------------------------------------------------------------

def fetch_enriched_rows(limit: int) -> list[dict]:
    """pipeline_status='enriched'인 raw_articles와 enrichment를 조인해 가져온다."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT e.id AS enrichment_id, e.raw_article_id, e.raw_category, e.raw_domain,
                      e.raw_entity, e.raw_event
               FROM enrichment e
               JOIN raw_articles a ON a.id = e.raw_article_id
               WHERE a.pipeline_status = 'enriched'
               ORDER BY e.id LIMIT %s""",
            (limit,),
        )
        return cur.fetchall()


def update_enrichment_normalized(
    enrichment_id: int,
    raw_article_id: int,
    category: str | None,
    domain: list[str],
    entity: list[str],
    event: list[str],
    normalization_method: str,
    synonym_table_version: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE enrichment
               SET category = %s, domain = %s, entity = %s, event = %s,
                   normalization_method = %s, synonym_table_version = %s
               WHERE id = %s""",
            (category, Jsonb(domain), Jsonb(entity), Jsonb(event), normalization_method,
             synonym_table_version, enrichment_id),
        )
        conn.execute(
            "UPDATE raw_articles SET pipeline_status = 'normalized' WHERE id = %s", (raw_article_id,)
        )


# ---------------------------------------------------------------------------
# synonym_table
# ---------------------------------------------------------------------------

def fetch_synonym_entries(dimension: str) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, canonical_value, aliases FROM synonym_table WHERE dimension = %s",
            (dimension,),
        )
        return cur.fetchall()


def upsert_synonym_entry(dimension: str, canonical_value: str, aliases: list[str],
                          *, reviewed: bool = False) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO synonym_table (dimension, canonical_value, aliases, reviewed)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (dimension, canonical_value)
               DO UPDATE SET aliases = %s, updated_at = now()""",
            (dimension, canonical_value, Jsonb(aliases), reviewed, Jsonb(aliases)),
        )


def fetch_raw_value_counts(dimension: str) -> list[tuple[str, int]]:
    """synonym_builder.py가 클러스터링할 원시값과 등장 빈도를 dimension별로 가져온다.

    빈도는 클러스터 안에서 canonical_value(가장 자주 쓰인 표현)를 고르는 데 쓰인다.
    """
    with get_conn() as conn, conn.cursor() as cur:
        if dimension in _ARRAY_DIMENSIONS:
            column = _ARRAY_DIMENSIONS[dimension]
            cur.execute(
                f"SELECT val, COUNT(*) AS cnt FROM ("
                f"  SELECT jsonb_array_elements_text({column}) AS val"
                f"  FROM enrichment WHERE {column} IS NOT NULL"
                f") sub GROUP BY val"
            )
        elif dimension in _SCALAR_DIMENSIONS:
            column = _SCALAR_DIMENSIONS[dimension]
            cur.execute(
                f"SELECT {column} AS val, COUNT(*) AS cnt FROM enrichment "
                f"WHERE {column} IS NOT NULL GROUP BY {column}"
            )
        else:
            raise ValueError(f"알 수 없는 dimension: {dimension}")
        return [(row["val"], row["cnt"]) for row in cur.fetchall() if row["val"]]
