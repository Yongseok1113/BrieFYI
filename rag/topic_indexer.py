"""기존 기사에 Category/Domain/Entity prototype metadata를 저장한다."""
import argparse
import json
from collections.abc import Iterable

from db.db import get_conn, init_db
from tools.topic_extract import extract_article_topics


def _load_articles(article_ids: list[int]) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id, title, description
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()


def _normalize_external_topics(category: str, domains: Iterable[str]) -> tuple[str, list[str]]:
    category = category.strip()
    if not category:
        raise ValueError("category는 비어 있을 수 없습니다.")

    normalized_domains = list(
        dict.fromkeys(domain.strip() for domain in domains if domain.strip())
    )
    if not normalized_domains:
        raise ValueError("domains는 하나 이상의 값이 필요합니다.")
    return category, normalized_domains


def save_article_topics(
    article_id: int,
    category: str,
    domains: list[str],
    entities: list[str],
    *,
    conn=None,
) -> None:
    """기사의 Category/Domain/Entity만 article_id 기준으로 UPSERT한다."""
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


def index_article_topics(
    article_ids: Iterable[int],
    category: str,
    domains: Iterable[str],
) -> list[dict]:
    """명시된 기존 기사에 외부 Category/Domain과 추출한 Entity를 저장한다."""
    requested_ids = list(dict.fromkeys(int(article_id) for article_id in article_ids))
    if not requested_ids:
        return []

    category, normalized_domains = _normalize_external_topics(category, domains)
    articles = _load_articles(requested_ids)
    found_ids = {article["id"] for article in articles}
    missing_ids = [article_id for article_id in requested_ids if article_id not in found_ids]
    if missing_ids:
        raise ValueError(f"존재하지 않는 article_id: {missing_ids}")

    results: list[dict] = []
    for article in articles:
        try:
            extracted = extract_article_topics(article["title"], article["description"])
            save_article_topics(
                article_id=article["id"],
                category=category,
                domains=normalized_domains,
                entities=extracted["entities"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"article_id={article['id']}의 topic indexing에 실패했습니다."
            ) from exc

        results.append(
            {
                "article_id": article["id"],
                "title": article["title"],
                "category": category,
                "domains": normalized_domains,
                "entities": extracted["entities"],
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기존 기사 4-Layer metadata indexing")
    parser.add_argument("--article-ids", nargs="+", type=int, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        required=True,
        help="여러 domain은 --domain을 반복해서 지정",
    )
    args = parser.parse_args(argv)

    init_db()
    results = index_article_topics(args.article_ids, args.category, args.domains)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
