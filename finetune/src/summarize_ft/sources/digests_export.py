"""자체 파이프라인(digests + raw_articles 테이블)에서 학습 데이터를 export한다.

design doc 3.1절: (원문 기사, Claude가 만든 요약/인사이트) 쌍을 지식 증류 관점의
1차 데이터로 쓴다. Claude를 교사 모델, 새 오픈모델을 학생 모델로 보는 구조.

repo 루트의 db/db.py, config.py를 그대로 재사용한다(스키마 중복 정의 방지).
finetune/은 별도 src 레이아웃이라 repo 루트를 sys.path에 추가해야 import된다.

task별로 두 종류의 예제를 만든다.
  - summarize: 각 topic-summary 항목마다 하나씩. input.article_text는 그 topic의
    source_urls에 해당하는 raw_articles의 title+description을 이어붙인 것 —
    "그 요약이 실제로 근거한 기사만" 입력으로 삼아야 완성 후 grounding이 맞는다.
  - insight: digests 한 행당 하나. input.summaries = summary_json 그대로,
    output = insight_json 그대로 (tools/insight.py 출력 스키마와 100% 동일).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fetch_rows(min_digest_date: str | None = None) -> list[dict[str, Any]]:
    """digests 테이블 전체(또는 min_digest_date 이후)를 dict 리스트로 가져온다."""
    from db.db import get_conn  # noqa: E402  (repo 루트 import, sys.path 조작 이후에만 가능)

    query = "SELECT id, digest_date, keyword, summary_json, insight_json FROM digests"
    params: tuple = ()
    if min_digest_date:
        query += " WHERE digest_date >= %s"
        params = (min_digest_date,)
    query += " ORDER BY digest_date"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def _fetch_articles_by_digest_date(digest_date) -> list[dict[str, Any]]:
    from db.db import get_conn  # noqa: E402

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title, description, url FROM raw_articles WHERE digest_date = %s",
            (digest_date,),
        )
        return cur.fetchall()


def _build_summarize_examples(digest_row: dict, articles_by_url: dict[str, dict]) -> list[dict]:
    examples = []
    topics = digest_row["summary_json"] or []
    for topic in topics:
        source_urls = topic.get("source_urls", [])
        matched = [articles_by_url[u] for u in source_urls if u in articles_by_url]
        if not matched:
            continue  # 근거 기사를 못 찾으면 학습 데이터로 못 씀 (grounding 불가)

        article_text = "\n\n".join(
            f"제목: {a['title']}\n설명: {a.get('description') or ''}" for a in matched
        )
        examples.append(
            {
                "id": str(uuid.uuid4()),
                "task": "summarize",
                "source": "digest_pipeline",
                "input": {
                    "article_title": topic.get("topic_title", ""),
                    "article_text": article_text,
                    "prompt_template": "summarize_v1",
                },
                "output": {
                    "topic_title": topic.get("topic_title", ""),
                    "summary": topic.get("summary", ""),
                    "source_urls": source_urls,
                },
                "meta": {
                    "created_at": str(digest_row["digest_date"]),
                    "teacher_model": "claude",  # 정확한 모델명은 config.ANTHROPIC_MODEL 참고, 실행 시점에 바뀔 수 있어 고정 문자열만 남김
                    "quality_flag": "unverified",  # 3.3절: grounding_check 통과분만 verified로 승격
                },
            }
        )
    return examples


def _build_insight_example(digest_row: dict) -> dict | None:
    summaries = digest_row["summary_json"]
    insight = digest_row["insight_json"]
    if not summaries or not insight or not insight.get("insights"):
        return None
    return {
        "id": str(uuid.uuid4()),
        "task": "insight",
        "source": "digest_pipeline",
        "input": {"summaries": summaries, "prompt_template": "insight_v1"},
        "output": insight,
        "meta": {
            "created_at": str(digest_row["digest_date"]),
            "teacher_model": "claude",
            "quality_flag": "unverified",
        },
    }


def export_examples(min_digest_date: str | None = None) -> list[dict]:
    """digests 테이블 전체를 공통 스키마 예제 리스트로 변환한다.

    scripts/prepare_data.py가 이 함수를 호출해 data/processed/에 합친다.
    """
    from .. import schema  # 순환 import 방지를 위해 지연 import

    rows = _fetch_rows(min_digest_date)
    all_examples: list[dict] = []

    articles_cache: dict[Any, dict[str, dict]] = {}

    for row in rows:
        digest_date = row["digest_date"]
        if digest_date not in articles_cache:
            articles = _fetch_articles_by_digest_date(digest_date)
            articles_cache[digest_date] = {a["url"]: a for a in articles}
        articles_by_url = articles_cache[digest_date]

        all_examples.extend(_build_summarize_examples(row, articles_by_url))

        insight_example = _build_insight_example(row)
        if insight_example:
            all_examples.append(insight_example)

    for ex in all_examples:
        schema.validate_example(ex, strict_id=True)

    return all_examples


if __name__ == "__main__":
    import argparse

    from ..jsonl import write_jsonl

    parser = argparse.ArgumentParser(description="digests 테이블 -> 공통 스키마 JSONL export")
    parser.add_argument("--out", default="finetune/data/processed/digest_pipeline.jsonl")
    parser.add_argument("--since", default=None, help="이 날짜(YYYY-MM-DD) 이후 digest만 export")
    args = parser.parse_args()

    examples = export_examples(min_digest_date=args.since)
    n = write_jsonl(args.out, examples)
    print(f"{n}건 export 완료 -> {args.out}")
