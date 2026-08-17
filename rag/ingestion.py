"""GNews 수집부터 GLiNER2 4-Layer/RAG indexing까지 실행하는 독립 worker."""
import argparse
import json
from datetime import date

from db.db import get_conn, init_db, insert_articles
from tools.news_fetch import fetch_news

from .indexer import index_articles


def _article_ids_for_urls(urls: list[str]) -> list[int]:
    if not urls:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, url
               FROM raw_articles
               WHERE url = ANY(%s)
               ORDER BY id""",
            (urls,),
        ).fetchall()
    return [row["id"] for row in rows]


def collect_and_index(
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
    existing_ids = set(_article_ids_for_urls(urls))
    inserted_count = insert_articles(date.today().isoformat(), articles)
    article_ids = [
        article_id
        for article_id in _article_ids_for_urls(urls)
        if article_id not in existing_ids
    ]
    indexed = index_articles(article_ids, device=device)
    return {
        "fetched_count": len(articles),
        "inserted_count": inserted_count,
        "article_ids": article_ids,
        "indexed": indexed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GNews 수집 + GLiNER2 4-Layer RAG indexing")
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)

    init_db()
    result = collect_and_index(
        args.keyword,
        args.days,
        args.max_results,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
