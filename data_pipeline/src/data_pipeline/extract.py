"""변형1의 뒷부분: 'pending' 상태 raw_articles에 로컬 키워드를 채우고 'extracted'로 전환한다.

이 단계는 API 호출이 아니라 요청 제한과 무관하다. main.py(라이브 파이프라인)가 넣은
기존 raw_articles 행에도 그대로 적용되므로, data_pipeline 없이 이미 쌓여있던 데이터를
백필하는 용도로도 쓸 수 있다.
"""
from __future__ import annotations

from . import db
from .keywords import extract_keywords


def run_extract(limit: int) -> dict:
    pending = db.fetch_articles_by_status("pending", limit)
    done, failed = 0, 0

    for article in pending:
        try:
            text = f"{article['title']} {article.get('description') or ''}"
            keywords = extract_keywords(text)
            db.set_article_keywords(article["id"], keywords)
            done += 1
        except Exception as exc:  # noqa: BLE001
            db.mark_failed(article["id"], "extract", str(exc))
            failed += 1

    return {"stage": "extract", "candidates": len(pending), "done": done, "failed": failed}
