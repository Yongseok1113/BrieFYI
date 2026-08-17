"""기사 본문, BGE chunk/embedding, GLiNER2 4-Layer를 함께 저장한다."""
import argparse
import json
from collections.abc import Iterable

from pgvector import Vector
from pgvector.psycopg import register_vector

from config import config
from db.db import get_conn, init_db
from tools.article_content import ArticleContentDependencyError, fetch_article_body

from .chunk import build_article_text, load_embedding_tokenizer, split_text
from .embed import embed_texts
from .event_extractor import GLiNER2EventExtractor, load_event_extractor
from .event_indexer import index_event_articles
from .topic_extractor import GLiNER2TopicExtractor
from .topic_indexer import save_article_topics


def _load_articles(article_ids: list[int]) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id, title, description, url
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()


def _resolve_article_text(article: dict) -> tuple[str, str]:
    """URL 본문을 우선하고 어떤 수집 실패든 metadata fallback으로 격리한다."""
    try:
        body = fetch_article_body(article["url"])
    except ArticleContentDependencyError:
        raise
    except Exception:  # 뉴스 사이트별 차단/HTML 차이는 해당 기사 fallback으로 끝낸다.
        body = None

    text_source = "body" if body else "title+description"
    text = build_article_text(article["title"], article["description"], body)
    return text, text_source


def _store_article(
    article: dict,
    chunks: list[dict],
    vectors: list[list[float]],
    text_source: str,
    topics: dict,
) -> dict:
    with get_conn() as conn:
        register_vector(conn)
        existing_rows = conn.execute(
            """SELECT id, chunk_index, chunk_text
               FROM article_chunks
               WHERE article_id = %s
               ORDER BY chunk_index""",
            (article["id"],),
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
                (article["id"],),
            )
            conn.execute(
                "DELETE FROM article_events WHERE article_id = %s",
                (article["id"],),
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
                (article["id"], chunk["chunk_index"], chunk["chunk_text"]),
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
            (article["id"], current_indexes),
        )
        save_article_topics(
            article["id"],
            topics["category"],
            topics["domains"],
            topics["entities"],
            conn=conn,
        )

    return {
        "article_id": article["id"],
        "chunk_count": len(chunks),
        "text_source": text_source,
        "category": topics["category"],
        "domains": topics["domains"],
        "entities": topics["entities"],
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "embedding_dimension": config.HF_EMBEDDING_DIMENSION,
    }


def index_articles(
    article_ids: Iterable[int],
    *,
    device: str = "cuda",
) -> list[dict]:
    """명시한 기사의 본문·4-Layer metadata·embedding을 함께 인덱싱한다."""
    requested_ids = list(dict.fromkeys(int(article_id) for article_id in article_ids))
    if not requested_ids:
        return []

    articles = _load_articles(requested_ids)
    found_ids = {article["id"] for article in articles}
    missing_ids = [article_id for article_id in requested_ids if article_id not in found_ids]
    if missing_ids:
        raise ValueError(f"존재하지 않는 article_id: {missing_ids}")

    tokenizer = load_embedding_tokenizer()
    event_extractor = load_event_extractor(device)
    topic_extractor = GLiNER2TopicExtractor(event_extractor.model)
    prepared: list[tuple[dict, list[dict], str, dict]] = []
    all_texts: list[str] = []
    for article in articles:
        text, text_source = _resolve_article_text(article)
        chunks = split_text(text, tokenizer=tokenizer)
        if not chunks:
            raise ValueError(f"article_id={article['id']}의 indexing 텍스트가 비어 있습니다.")
        topics = topic_extractor.extract(text)
        prepared.append((article, chunks, text_source, topics))
        all_texts.extend(chunk["chunk_text"] for chunk in chunks)

    all_vectors = embed_texts(all_texts)
    results: list[dict] = []
    offset = 0
    for article, chunks, text_source, topics in prepared:
        end = offset + len(chunks)
        results.append(
            _store_article(
                article,
                chunks,
                all_vectors[offset:end],
                text_source,
                topics,
            )
        )
        offset = end
    event_results = index_event_articles(
        [result["article_id"] for result in results],
        device=device,
        extractor=GLiNER2EventExtractor(event_extractor.model),
    )
    events_by_article = {result["article_id"]: result for result in event_results}
    for result in results:
        event_result = events_by_article[result["article_id"]]
        result["event_status"] = event_result["status"]
        result["event_count"] = event_result["event_count"]
        if event_result.get("error"):
            result["event_error"] = event_result["error"]
    return results


def index_article(article_id: int, *, device: str = "cuda") -> dict:
    """기사 한 건의 URL 본문을 다시 읽어 인덱싱한다."""
    return index_articles([article_id], device=device)[0]


def index_all_articles(limit: int = 20, *, device: str = "cuda") -> list[dict]:
    """embedding이 없는 기사 중 제한된 수만 본문 수집과 인덱싱을 수행한다."""
    if limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")

    with get_conn() as conn:
        article_ids = [
            row["id"]
            for row in conn.execute(
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
            )
        ]
    return index_articles(article_ids, device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BrieFYI 기사 본문 RAG indexing worker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--article-ids", nargs="+", type=int)
    group.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)

    init_db()
    if args.article_ids is not None:
        results = index_articles(args.article_ids, device=args.device)
    else:
        results = index_all_articles(
            args.limit if args.limit is not None else 20,
            device=args.device,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
