"""
RAG Indexer

기사 chunk를 임베딩하고
PostgreSQL + pgvector에 저장한다.
"""

from typing import List, Dict

import psycopg2

from rag.embed import embed_texts


# ==========================================
# 1. PostgreSQL 설정
# ==========================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_test",
    "user": "rag_user",
    "password": "rag_password",
}


# ==========================================
# 2. DB 연결
# ==========================================

def get_connection():
    """
    PostgreSQL 연결을 생성한다.
    """

    return psycopg2.connect(
        **DB_CONFIG
    )


# ==========================================
# 3. Chunk 저장
# ==========================================

def index_chunks(
    title: str,
    source_url: str,
    chunks: List[str]
) -> int:
    """
    기사 chunk들을 임베딩하여 DB에 저장한다.

    Args:
        title: 기사 제목
        source_url: 기사 원문 URL
        chunks: chunk 문자열 리스트

    Returns:
        저장된 chunk 개수
    """

    if not chunks:
        return 0

    # --------------------------------------
    # Chunk 전체를 한 번에 임베딩
    # --------------------------------------

    embeddings = embed_texts(chunks)

    conn = get_connection()
    cur = conn.cursor()

    saved_count = 0

    try:

        # ----------------------------------
        # Chunk별 DB 저장
        # ----------------------------------

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            cur.execute(
                """
                INSERT INTO rag_documents
                (
                    title,
                    content,
                    source_url,
                    embedding
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    title,
                    chunk,
                    source_url,
                    embedding
                )
            )

            saved_count += 1

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        cur.close()
        conn.close()

    return saved_count


# ==========================================
# 4. 테스트용 조회
# ==========================================

def get_document_count() -> int:
    """
    현재 rag_documents에 저장된 chunk 개수를 반환한다.
    """

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM rag_documents
            """
        )

        count = cur.fetchone()[0]

        return count

    finally:

        cur.close()
        conn.close()


# ==========================================
# 5. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG Indexer 테스트")
    print("=" * 60)

    test_title = "테스트 뉴스 기사"

    test_url = "https://example.com/test"

    test_chunks = [
        "프로야구 경기에서 새로운 기록이 나왔다.",
        "선수들은 이번 경기에서 좋은 활약을 보여주었다.",
    ]

    print("\nChunk 임베딩 및 DB 저장 중...")

    saved = index_chunks(
        title=test_title,
        source_url=test_url,
        chunks=test_chunks
    )

    print(f"저장된 Chunk: {saved}개")

    total = get_document_count()

    print(f"현재 DB 전체 Chunk: {total}개")

    print("\n" + "=" * 60)
    print("Indexer 테스트 완료")
    print("=" * 60)