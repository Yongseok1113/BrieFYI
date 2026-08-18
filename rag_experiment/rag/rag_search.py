"""
RAG 통합 검색

Retriever로 후보를 검색하고
Reranker로 최종 순위를 결정한다.
"""

from rag.retriever import retrieve_documents
from rag.reranker import rerank_documents


def search(
    query: str,
    retrieve_top_k: int = 10,
    rerank_top_k: int = 3
):
    """
    Retriever → Reranker 순서로 RAG 검색을 수행한다.

    1. Retriever가 pgvector에서 후보 문서를 검색한다.
    2. Embedding 모델을 메모리에서 해제한다.
    3. Reranker가 후보 문서를 재정렬한다.
    4. 최종 검색 결과를 반환한다.
    """

    print("\n" + "=" * 60)
    print("RAG 통합 검색")
    print("=" * 60)

    print(f"\n검색어: {query}")

    # ==========================================
    # 1. Retriever
    # ==========================================

    print("\n[1단계] Retriever 검색 중...")

    candidates = retrieve_documents(
        query=query,
        top_k=retrieve_top_k
    )

    print(f"Retriever 후보: {len(candidates)}개")


    if not candidates:
        print("검색 결과가 없습니다.")
        return []

    # ==========================================
    # 2. Reranker
    # ==========================================

    print("\n[2단계] Reranker 재정렬 중...")

    results = rerank_documents(
        query=query,
        documents=candidates,
        top_k=rerank_top_k
    )

    # ==========================================
    # 3. 최종 결과 출력
    # ==========================================

    print("\n" + "=" * 60)
    print("최종 검색 결과")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print("\n" + "-" * 60)
        print(f"[결과 {index}]")

        print(f"제목: {result.get('title', '')}")

        print(
            f"pgvector distance: "
            f"{result.get('distance', 0):.4f}"
        )

        print(
            f"Reranker score: "
            f"{result.get('reranker_score', 0):.4f}"
        )

        print(f"URL: {result.get('source_url', '')}")

        print(
            f"내용: "
            f"{result.get('content', '')[:300]}"
        )

    return results


# ==========================================
# 테스트
# ==========================================

if __name__ == "__main__":

    print("rag_search.py 실행 확인")

    search(
        query="프로야구 경기 관련 뉴스",
        retrieve_top_k=10,
        rerank_top_k=3
    )