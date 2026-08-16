"""
RAG Indexer

기사 chunk를 임베딩하고
PostgreSQL + pgvector에 저장한다.

[변경사항]
기존에는 title, content, source_url, embedding만 저장했는데,
이번에 category, domain, entities, event
4개 컬럼을 추가로 저장하도록 확장했다.
이 4개 값은 rag/classify.py가 미리 판단해서 넘겨준다.
(이 파일은 "저장"만 담당, "판단"은 classify.py가 담당 — 역할 분리)
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
    chunks: List[str],
    # --------------------------------------
    # 여기 4개가 새로 추가된 파라미터.
    #
    # = None 이 붙어있는 이유:
    #   "기본값이 None"이라는 뜻. 즉 이 4개를 안 넘겨도
    #   함수 호출 자체는 에러 없이 동작한다.
    #
    # str = None 같은 표기는 "타입 힌트"라고 해서
    #   "이 값은 보통 문자열이지만, 안 넘기면 None일 수도 있다"는
    #   설명용 표시일 뿐, 실제 동작에 영향은 없다.
    category: str = None,
    domain: str = None,
    entities: List[str] = None,
    event: str = None,
) -> int:
    """
    기사 chunk들을 임베딩하여 DB에 저장한다.

    Args:
        title: 기사 제목
        source_url: 기사 원문 URL
        chunks: chunk 문자열 리스트
        category: 대분류 (예: "기술") - classify.py가 판단한 값
        domain: 도메인 (예: "AI") - classify.py가 판단한 값
        entities: 엔티티 리스트 (예: ["엔비디아"]) - classify.py가 판단한 값
        event: 이벤트 (예: "투자") - classify.py가 판단한 값

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

            # ------------------------------
            # INSERT문에 category, domain, entities, event
            # 4개 컬럼을 추가했다.
            #
            # VALUES 뒤 괄호 안 %s 개수가
            # 컬럼 개수(8개)랑 정확히 일치해야 한다.
            #
            # entities는 파이썬 리스트(예: ["엔비디아"])를
            # 그대로 넘기면 psycopg2가 알아서
            # Postgres의 TEXT[] 배열 타입으로 변환해준다.
            # (별도로 문자열로 합치거나 하는 처리 필요 없음)
            cur.execute(
                """
                INSERT INTO rag_documents
                (
                    title,
                    content,
                    source_url,
                    embedding,
                    category,
                    domain,
                    entities,
                    event
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
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
                    embedding,
                    category,
                    domain,
                    entities,
                    event
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
# 기존과 동일 (안 바뀜)

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

    test_url = "https://example.com/test-metadata"

    test_chunks = [
        "엔비디아가 AI 반도체 스타트업에 대규모 투자를 단행했다.",
        "이번 투자로 AI 반도체 생태계가 더욱 확장될 전망이다.",
    ]

    # --------------------------------------
    # 이번 테스트에서는 category/domain/entities/event를
    # 직접 값으로 넣어서 테스트한다.
    # (실제 파이프라인에서는 sbs_to_rag.py가
    #  classify.py 결과를 여기 넣어주게 될 것 — 다음 파일에서 연결)
    print("\nChunk 임베딩 및 DB 저장 중 (metadata 포함)...")

    saved = index_chunks(
        title=test_title,
        source_url=test_url,
        chunks=test_chunks,
        category="기술",
        domain="AI",
        entities=["엔비디아"],
        event="투자",
    )

    print(f"저장된 Chunk: {saved}개")

    total = get_document_count()

    print(f"현재 DB 전체 Chunk: {total}개")

    print("\n" + "=" * 60)
    print("Indexer 테스트 완료 (metadata 저장 확인)")
    print("=" * 60)