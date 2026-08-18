"""
RAG Retriever

사용자의 검색어를 BGE-M3로 임베딩하고
PostgreSQL + pgvector에서 관련성이 높은
기사 Chunk 후보를 검색한다.

[변경사항]
category / domain / entity / event
4개 조건으로 먼저 거르고 벡터 검색을 하도록 확장했다.

예를 들어 domain="AI"를 넘기면,
전체 기사 중 domain이 정확히 "AI"인 것들만 후보로 남긴 뒤
그 안에서 벡터 유사도로 정렬한다.
"""

from typing import List, Dict, Any, Optional

import psycopg2

from rag.embed import embed_text

# --------------------------------------------------
# DB 접속 정보를 db_config.py에서 가져다 쓴다.
# (DB 정보를 하나로 통일)
from rag.db_config import DB_CONFIG


# ==========================================
# 1. DB 연결
# ==========================================
# 접속정보를 db_config.py에서 가져옴

def get_connection():
    """
    PostgreSQL 연결을 생성한다.
    """

    return psycopg2.connect(
        **DB_CONFIG
    )


# ==========================================
# 2. Vector 검색 (+ Metadata 필터)
# ==========================================

def retrieve_documents(
    query: str,
    top_k: int = 10,
    # --------------------------------------
    # 여기 4개가 새로 추가된 "필터용" 파라미터.
    #
    # Optional[str] = None
    #   → "문자열이거나 없을 수도 있다(None)"는 뜻.
    #     기본값이 None이라, 이 4개를 안 넘기면
    #     기존이랑 똑같이 필터 없는 검색으로 동작한다.
    category: Optional[str] = None,
    domain: Optional[str] = None,
    entity: Optional[str] = None,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    사용자의 검색어와 의미적으로 가까운
    기사 Chunk를 pgvector에서 검색한다.
    필요하면 category/domain/entity/event로
    먼저 후보를 좁힌 다음 검색한다.

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
                "published_at": ...,
                "keyword": ...,
                "category": "기술",
                "domain": "AI",
                "entities": ["NVIDIA"],
                "event": "투자",
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
        # 3. WHERE 조건 조립
        #
        # where_clauses: SQL의 WHERE 뒤에 들어갈 조건문 조각들
        #                (예: "category = %s") 을 담는 리스트
        # where_params:  그 조건문의 %s 자리에 실제로 들어갈 값들
        #                (예: "기술") 을 순서대로 담는 리스트
        #
        #   category만 넘겼을 수도 있고, domain+entity만
        #   넘겼을 수도 있고, 4개 다 넘겼을 수도 있어서
        #   상황마다 WHERE절 내용이 달라져야 하기 때문이다.
        #   그래서 "넘어온 것만" 조건에 추가하는 방식.
        # ----------------------------------

        where_clauses = []
        where_params = []

        # category가 넘어왔으면 (None이 아니면)
        # "category = %s" 조건을 추가하고,
        # 그 %s 자리에 넣을 실제 값(category)도 같이 기록해둔다.
        if category:
            where_clauses.append("category = %s")
            where_params.append(category)

        if domain:
            where_clauses.append("domain = %s")
            where_params.append(domain)

        # entities는 배열(TEXT[]) 컬럼이라서
        # "= " 이 아니라 "ANY(entities)" 라는 특별한 문법을 쓴다.
        # "%s = ANY(entities)" 뜻: "entities 배열 안에 %s 값이 있는가?"
        if entity:
            where_clauses.append("%s = ANY(entities)")
            where_params.append(entity)

        if event:
            where_clauses.append("event = %s")
            where_params.append(event)

        # --------------------------------------
        # where_clauses 리스트를 실제 SQL 문자열로 합친다.
        #
        # 만약 필터를 아무것도 안 넘겼으면 (where_clauses가 비어있으면)
        # where_sql은 그냥 빈 문자열("") 이 된다.
        # → 이 경우 SQL에서 WHERE절 자체가 사라지고,
        #   기존과 완전히 동일하게 "전체 기사 대상 벡터 검색"이 된다.
        where_sql = (
            "WHERE " + " AND ".join(where_clauses)
        ) if where_clauses else ""

        # --------------------------------------
        # 4. 최종 쿼리에 넘길 파라미터 순서 맞추기
        #
        # SQL문 안에서 %s가 나오는 순서와
        # 여기 params 리스트의 순서가 정확히 일치해야 한다.
        #
        # 순서: [검색어벡터] + [WHERE절 값들] + [검색어벡터(정렬용)] + [top_k]
        #       (SELECT의 distance 계산용 1번,
        #        WHERE절 조건들,
        #        ORDER BY의 distance 계산용 1번,
        #        LIMIT 값)
        params = (
            [query_embedding]
            + where_params
            + [query_embedding, top_k]
        )

        # ----------------------------------
        # 5. pgvector 유사도 검색 (+ 필터)
        #
        # f"""...{where_sql}...""" 처럼 만든 이유:
        #   where_sql이 "WHERE category = %s" 였다가
        #   빈 문자열("")이었다가 상황에 따라 바뀌기 때문에,
        #   그 자리에 그대로 끼워 넣는다.
        #
        # 주의: where_sql 안에는 사용자가 입력한 값이
        #       직접 들어가지 않는다 (전부 %s 자리표시자로 처리하고
        #       실제 값은 params로 따로 넘김).
        #       이렇게 해야 SQL 인젝션 같은 보안 문제를 피할 수 있다.
        # ----------------------------------

        cur.execute(
            f"""
            SELECT
                id,
                title,
                content,
                source_url,
                published_at,
                keyword,
                category,
                domain,
                entities,
                event,
                embedding <=> %s::vector AS distance
            FROM rag_documents
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        )

        rows = cur.fetchall()

        # ----------------------------------
        # 6. 결과를 Dictionary로 변환
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
                    "category": row[6],
                    "domain": row[7],
                    "entities": row[8],
                    "event": row[9],
                    "distance": float(row[10]),
                }
            )

        return documents

    finally:

        cur.close()
        conn.close()


# ==========================================
# 3. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG Retriever 테스트 (필터 없음)")
    print("=" * 60)

    query = "정치 관련 뉴스"

    print()
    print(f"검색어: {query}")
    print()
    print("pgvector 검색 중 (필터 없음)...")

    results = retrieve_documents(
        query=query,
        top_k=5,
    )

    print(f"\n검색된 후보: {len(results)}개\n")

    for index, result in enumerate(results, start=1):

        print("-" * 60)
        print(f"[후보 {index}]")
        print(f"제목: {result['title']}")
        print(f"category: {result['category']} / domain: {result['domain']} / event: {result['event']}")
        print(f"distance: {result['distance']:.4f}")

    # --------------------------------------
    # 필터를 넣어서 한 번 더 테스트
    # (현재 DB에는 정치 기사만 있어서 "기타"로 필터링해봄)
    # --------------------------------------

    print("\n" + "=" * 60)
    print("RAG Retriever 테스트 (category 필터 적용)")
    print("=" * 60)

    filtered_results = retrieve_documents(
        query=query,
        top_k=5,
        category="기타",
    )

    print(f"\ncategory='기타' 필터 검색된 후보: {len(filtered_results)}개\n")

    for index, result in enumerate(filtered_results, start=1):

        print("-" * 60)
        print(f"[후보 {index}]")
        print(f"제목: {result['title']}")
        print(f"category: {result['category']}")

    print()
    print("=" * 60)
    print("Retriever 테스트 완료")
    print("=" * 60)