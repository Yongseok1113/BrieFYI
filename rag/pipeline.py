"""GNews 수집부터 4-Layer indexing까지 이어 실행하는 오케스트레이션.

각 단계는 독립적으로도 실행 가능하다(cli.py). `run_all()`은 한 프로세스 안에서 수집과
indexing을 이어 돌리는 편의 함수로, 이번 실행에서 새로 저장된 기사만 인덱싱한다 —
기존 기사 backfill은 `indexer.index_all_articles()`나 명시적 ID 호출로 따로 처리한다.
"""
from __future__ import annotations

from datetime import date

from db.db import insert_articles
from tools.news_fetch import fetch_news

from . import db
from .indexer import index_articles


def run_all(
    keyword: str,
    lookback_days: int = 1,
    max_results: int = 10,
    *,
    device: str = "cuda",
) -> dict:
    """GNews 결과를 저장하고 이번 수집 기사들을 즉시 4-Layer indexing한다."""
    articles = fetch_news(keyword, lookback_days, max_results)
    urls = list(
        dict.fromkeys(article.get("url", "") for article in articles if article.get("url"))
    )
    existing_ids = set(db.load_article_ids_by_urls(urls))
    inserted_count = insert_articles(date.today().isoformat(), articles)
    article_ids = [
        article_id
        for article_id in db.load_article_ids_by_urls(urls)
        if article_id not in existing_ids
    ]
    indexed = index_articles(article_ids, device=device)
    return {
        "fetched_count": len(articles),
        "inserted_count": inserted_count,
        "article_ids": article_ids,
        "indexed": indexed,
    }
