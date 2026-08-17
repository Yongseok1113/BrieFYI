"""현재 raw_articles를 청킹하고 모델별 embedding을 저장한다."""
from collections.abc import Iterable

from pgvector import Vector
from pgvector.psycopg import register_vector

from config import config
from db.db import get_conn

from .chunk import build_article_text, split_text
from .embed import embed_texts


def _load_articles(article_ids: list[int]) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id, title, description
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()


def _store_article(article: dict, chunks: list[dict], vectors: list[list[float]]) -> dict:
    with get_conn() as conn:
        register_vector(conn)
        current_indexes: list[int] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            existing = conn.execute(
                """SELECT id, chunk_text FROM article_chunks
                   WHERE article_id = %s AND chunk_index = %s""",
                (article["id"], chunk["chunk_index"]),
            ).fetchone()
            if existing is not None and existing["chunk_text"] != chunk["chunk_text"]:
                # 같은 chunk_id에 저장된 다른 모델 embedding도 더 이상 유효하지 않다.
                conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = %s", (existing["id"],))

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

        if current_indexes:
            conn.execute(
                """DELETE FROM article_chunks
                   WHERE article_id = %s AND NOT (chunk_index = ANY(%s))""",
                (article["id"], current_indexes),
            )

    return {
        "article_id": article["id"],
        "chunk_count": len(chunks),
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "embedding_dimension": config.HF_EMBEDDING_DIMENSION,
    }


def index_articles(article_ids: Iterable[int]) -> list[dict]:
    """지정한 기사들을 한 번에 embedding한 뒤 기사별 transaction으로 저장한다."""
    requested_ids = list(dict.fromkeys(int(article_id) for article_id in article_ids))
    if not requested_ids:
        return []

    articles = _load_articles(requested_ids)
    found_ids = {article["id"] for article in articles}
    missing_ids = [article_id for article_id in requested_ids if article_id not in found_ids]
    if missing_ids:
        raise ValueError(f"존재하지 않는 article_id: {missing_ids}")

    prepared: list[tuple[dict, list[dict]]] = []
    all_texts: list[str] = []
    for article in articles:
        text = build_article_text(article["title"], article["description"])
        chunks = split_text(text)
        if not chunks:
            raise ValueError(f"article_id={article['id']}의 indexing 텍스트가 비어 있습니다.")
        prepared.append((article, chunks))
        all_texts.extend(chunk["chunk_text"] for chunk in chunks)

    all_vectors = embed_texts(all_texts)
    results: list[dict] = []
    offset = 0
    for article, chunks in prepared:
        end = offset + len(chunks)
        results.append(_store_article(article, chunks, all_vectors[offset:end]))
        offset = end
    return results


def index_article(article_id: int) -> dict:
    """기사 한 건을 인덱싱한다."""
    return index_articles([article_id])[0]


def index_all_articles() -> list[dict]:
    """아직 embedding이 없는 기사만 인덱싱한다."""
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
                   ORDER BY ra.id"""
            )
        ]
    return index_articles(article_ids)
