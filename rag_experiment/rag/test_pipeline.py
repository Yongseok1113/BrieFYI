"""
RAG 전체 파이프라인 테스트

Chunk
→ Embedding
→ Indexer
→ Retriever
→ Reranker
전체 과정이 정상적으로 연결되는지 확인한다.
"""

from rag.chunk import chunk_text
from rag.embed import embed_text
from rag.indexer import index_chunks
from rag.retriever import retrieve_documents
from rag.reranker import rerank_documents


def main():

    print("=" * 70)
    print("RAG 전체 파이프라인 테스트")
    print("=" * 70)

    # ==========================================
    # 1. Chunk
    # ==========================================

    print("\n[1/5] Chunk 테스트")

    text = """
    프로야구 경기에서 새로운 기록이 나왔다.
    선수들은 이번 경기에서 좋은 활약을 보여주었다.
    이번 시즌 프로야구는 많은 관심을 받고 있다.
    """

    chunks = chunk_text(text)

    print(f"  생성된 Chunk: {len(chunks)}개")

    if not chunks:
        raise RuntimeError("Chunk 생성 실패")

    print("  ✓ Chunk 정상")

    # ==========================================
    # 2. Embedding
    # ==========================================

    print("\n[2/5] Embedding 테스트")

    embedding = embed_text(chunks[0])

    print(f"  벡터 차원: {len(embedding)}")

    if len(embedding) != 1024:
        raise RuntimeError(
            f"Embedding 차원 오류: {len(embedding)}"
        )

    print("  ✓ Embedding 정상")

    # ==========================================
    # 3. Indexer
    # ==========================================

    print("\n[3/5] Indexer 테스트")

    saved = index_chunks(
        title="RAG 전체 테스트 뉴스",
        source_url="https://example.com/rag-test",
        chunks=chunks
    )

    print(f"  DB 저장 Chunk: {saved}개")

    if saved != len(chunks):
        raise RuntimeError(
            f"Indexer 저장 개수 불일치: "
            f"{saved} / {len(chunks)}"
        )

    print("  ✓ Indexer 정상")

    # ==========================================
    # 4. Retriever
    # ==========================================

    print("\n[4/5] Retriever 테스트")

    query = "프로야구 경기와 관련된 뉴스"

    candidates = retrieve_documents(
        query=query,
        top_k=5
    )

    print(f"  Retriever 후보: {len(candidates)}개")

    if not candidates:
        raise RuntimeError("Retriever 검색 결과 없음")

    for i, document in enumerate(candidates, 1):

        print(
            f"    [{i}] "
            f"{document.get('title', '')} "
            f"(distance="
            f"{document.get('distance', 0):.4f})"
        )

    print("  ✓ Retriever 정상")

    # ==========================================
    # 5. Reranker
    # ==========================================

    print("\n[5/5] Reranker 테스트")

    results = rerank_documents(
        query=query,
        documents=candidates,
        top_k=3
    )

    print(f"  최종 결과: {len(results)}개")

    if not results:
        raise RuntimeError("Reranker 결과 없음")

    for i, document in enumerate(results, 1):

        print(
            f"    [{i}] "
            f"{document.get('title', '')} "
            f"(score="
            f"{document.get('reranker_score', 0):.4f})"
        )

    print("  ✓ Reranker 정상")

    # ==========================================
    # 최종 결과
    # ==========================================

    print("\n" + "=" * 70)
    print("RAG 전체 파이프라인 테스트 완료")
    print("=" * 70)

    print("""
✓ Chunk
✓ Embedding (1024차원)
✓ Indexer
✓ Retriever
✓ BGE Reranker

→ RAG 핵심 모듈 연결 정상
""")


if __name__ == "__main__":
    main()
