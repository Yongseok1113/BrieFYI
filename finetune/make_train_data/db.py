"""레포 루트 db.db.get_conn()을 재사용해 raw_articles(+enrichment)를 조회한다.
LEFT JOIN이다 — enrichment_export.py(INNER JOIN + pipeline_status='normalized'만)와
달리, 클러스터링은 enrichment 유무와 무관하게 모든 raw_articles를 대상으로 하고
enrichment 유무만 entity_extract.py의 분기 신호로 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def fetch_articles(since: str | None = None) -> list[dict[str, Any]]:
    from db.db import get_conn

    query = """
        SELECT a.id, a.title, a.description, a.url, a.source, a.published_at,
               e.category, e.domain, e.entity, e.event, e.insights
        FROM raw_articles a
        LEFT JOIN enrichment e ON e.raw_article_id = a.id
    """
    params: tuple = ()
    if since:
        query += " WHERE a.published_at >= %s"
        params = (since,)
    query += " ORDER BY a.published_at"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
