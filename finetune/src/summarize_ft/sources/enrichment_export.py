"""data_pipeline/(변형1~3)이 만든 raw_articles+enrichment에서 학습 데이터를 export한다.

docs/data-pipeline-design.md 9절: raw_articles(변형1 결과) + enrichment(변형3 결과)를
조인해 schema.py의 "enrich" task 스키마로 변환한다. digests_export.py와 마찬가지로
repo 루트의 db.py/config.py를 재사용한다(스키마 중복 정의 방지) — finetune/은 같은 레포
체크아웃 안에서 스크립트로 실행되는 걸 전제하므로 sys.path 트릭이 안전하다(data_pipeline은
반대로 별도 컨테이너라 이 트릭을 안 쓰고 자체 db.py를 둔 것과 대비된다, 8절 참고).

pipeline_status='normalized'인 행만 export한다 — 정규화(변형3)까지 끝난 것만 학습 데이터로
채택한다는 뜻이다.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fetch_rows(min_id: int | None = None) -> list[dict[str, Any]]:
    from db.db import get_conn  # noqa: E402  (repo 루트 import, sys.path 조작 이후에만 가능)

    query = """
        SELECT a.id AS raw_article_id, a.title, a.description, a.url,
               e.insights, e.implications, e.category, e.domain, e.entity, e.event,
               e.model_used, e.prompt_version, e.normalization_method
        FROM raw_articles a
        JOIN enrichment e ON e.raw_article_id = a.id
        WHERE a.pipeline_status = 'normalized'
    """
    params: tuple = ()
    if min_id:
        query += " AND a.id >= %s"
        params = (min_id,)
    query += " ORDER BY a.id"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def _to_example(row: dict[str, Any]) -> dict[str, Any] | None:
    insights = row.get("insights") or []
    if not insights:
        return None  # schema.py enrich task는 insights 3~5개를 요구 — 없으면 학습 데이터로 못 씀

    return {
        "id": str(uuid.uuid4()),
        "task": "enrich",
        "source": "data_pipeline",
        "input": {
            "article_title": row["title"],
            "article_text": f"제목: {row['title']}\n설명: {row.get('description') or ''}",
            "prompt_template": row.get("prompt_version") or "enrich_v1",
        },
        "output": {
            "insights": insights,
            "implications": row.get("implications") or [],
            "category": row.get("category") or "",
            "domain": row.get("domain") or [],
            "entity": row.get("entity") or [],
            "event": row.get("event") or [],
        },
        "meta": {
            "created_at": "",
            "teacher_model": row.get("model_used") or "",
            "quality_flag": "verified" if row.get("normalization_method") == "exact" else "unverified",
            "raw_article_id": row["raw_article_id"],
        },
    }


def export_examples(min_id: int | None = None) -> list[dict]:
    """raw_articles+enrichment -> 공통 스키마 예제 리스트. scripts/prepare_data.py가 호출한다."""
    from .. import schema  # 순환 import 방지를 위해 지연 import

    rows = _fetch_rows(min_id)
    examples = [ex for row in rows if (ex := _to_example(row)) is not None]

    for ex in examples:
        schema.validate_example(ex, strict_id=True)

    return examples


if __name__ == "__main__":
    import argparse

    from ..jsonl import write_jsonl

    parser = argparse.ArgumentParser(description="raw_articles+enrichment -> 공통 스키마 JSONL export")
    parser.add_argument("--out", default="finetune/data/processed/enrichment.jsonl")
    parser.add_argument("--min-id", type=int, default=None, help="이 raw_article id 이상만 export")
    args = parser.parse_args()

    examples = export_examples(min_id=args.min_id)
    n = write_jsonl(args.out, examples)
    print(f"{n}건 export 완료 -> {args.out}")
