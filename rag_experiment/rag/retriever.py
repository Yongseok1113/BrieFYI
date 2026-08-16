"""
RAG Retriever

사용자의 검색어를 BGE-M3로 임베딩하고
PostgreSQL + pgvector에서 관련성이 높은
기사 Chunk 후보를 검색한다.

Retriever의 역할:
    검색어
      ↓
    BGE-M3 Embedding
      ↓
    pgvector 유사도 검색
      ↓
    후보 기사 반환

Reranking은 reranker.py에서 담당한다.
"""

from typing import List, Dict, Any

import psycopg2

from rag.embed import embed_text


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
# 3. Vector 검색
# ==========================================

def retrieve_documents(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    사용자의 검색어와 의미적으로 가까운
    기사 Chunk를 pgvector에서 검색한다.

    Args:
        query:
            사용자 검색어

        top_k:
            가져올 후보 Chunk 개수

    Returns:
        검색된 문서 목록

        예:
        [
            {
                "id": 1,
                "title": "...",
                "content": "...",
                "source_url": "...",
                "distance": 0.1234
            }
        ]
    """

    if not query or not query.strip():
        return []

    # --------------------------------------
    # 1. 검색어 Embedding
    # --------------------------------------

    query_embedding = embed_text(query)

    # --------------------------------------
    # 2. PostgreSQL 연결
    # --------------------------------------

    conn = get_connection()
    cur = conn.cursor()

    try:

        # ----------------------------------
        # 3. pgvector 유사도 검색
        # ----------------------------------

        cur.execute(
            """
            SELECT
                id,
                title,
                content,
                source_url,
                published_at,
                keyword,
                embedding <=> %s::vector AS distance
            FROM rag_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        )

        rows = cur.fetchall()

        # ----------------------------------
        # 4. 결과를 Dictionary로 변환
        # ----------------------------------

        documents = []

        for row in rows:

            documents.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "source_url": row[3],
                    "published_at": row[4],
                    "keyword": row[5],
                    "distance": float(row[6]),
                }
            )

        return documents

    finally:

        cur.close()
        conn.close()


# ==========================================
# 4. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG Retriever 테스트")
    print("=" * 60)

    query = "프로야구 경기 관련 뉴스"

    print()
    print(f"검색어: {query}")
    print()
    print("pgvector 검색 중...")

    results = retrieve_documents(
        query=query,
        top_k=10,
    )

    print()
    print(f"검색된 후보: {len(results)}개")

    print()

    for index, result in enumerate(
        results,
        start=1,
    ):

        print("-" * 60)

        print(f"[후보 {index}]")

        print(f"ID: {result['id']}")

        print(f"제목: {result['title']}")

        print(
            f"pgvector distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"URL: "
            f"{result['source_url']}"
        )

    print()
    print("=" * 60)
    print("Retriever 테스트 완료")
    print("=" * 60)