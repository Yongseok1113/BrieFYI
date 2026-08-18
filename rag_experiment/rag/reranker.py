"""
RAG Reranker

Retriever가 가져온 후보 문서를
BAAI/bge-reranker-v2-m3 모델로 재평가하여
검색어와 가장 관련성이 높은 문서 순으로 정렬한다.
"""

from typing import List, Dict, Any

from FlagEmbedding import FlagReranker


# ==========================================
# 1. Reranker 모델 설정
# ==========================================

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


# ==========================================
# 2. Reranker 모델
# ==========================================

_reranker = None


def get_reranker():
    """
    필요할 때만 Reranker 모델을 로딩한다.

    파일을 import하는 순간에는 모델을 로딩하지 않는다.
    """

    global _reranker

    if _reranker is None:

        print()
        print("=" * 60)
        print("BGE Reranker 모델 로딩 중...")
        print(f"모델: {MODEL_NAME}")
        print("=" * 60)

        _reranker = FlagReranker(
            MODEL_NAME,
            use_fp16=False
        )

        print("Reranker 모델 로딩 완료")

    return _reranker


# ==========================================
# 3. Reranking
# ==========================================

def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retriever가 가져온 후보 문서를
    BGE Reranker로 재평가한다.

    query:
        사용자의 검색어

    documents:
        Retriever가 반환한 후보 문서

    top_k:
        최종 반환할 문서 개수
    """

    if not documents:
        return []

    # 모델은 실제 함수가 호출될 때 로딩
    reranker = get_reranker()

    # --------------------------------------
    # Query + Document 쌍 생성
    # --------------------------------------

    pairs = []

    for document in documents:

        title = document.get("title", "")
        content = document.get("content", "")

        document_text = (
            f"{title}\n"
            f"{content}"
        )

        pairs.append(
            [query, document_text]
        )

    # --------------------------------------
    # Reranker 점수 계산
    # --------------------------------------

    scores = reranker.compute_score(
        pairs,
        normalize=True
    )

    # 문서가 하나일 경우 float가 반환될 수 있음
    if isinstance(scores, float):
        scores = [scores]

    # --------------------------------------
    # 점수 추가
    # --------------------------------------

    reranked_documents = []

    for document, score in zip(
        documents,
        scores
    ):

        result = document.copy()

        result["reranker_score"] = float(score)

        reranked_documents.append(result)

    # --------------------------------------
    # 높은 점수 순으로 정렬
    # --------------------------------------

    reranked_documents.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # --------------------------------------
    # Top K 반환
    # --------------------------------------

    return reranked_documents[:top_k]


# ==========================================
# 4. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("BGE Reranker 테스트")
    print("=" * 60)

    query = "프로야구 경기 관련 뉴스"

    test_documents = [

        {
            "title": "프로야구 새로운 기록",
            "content": (
                "프로야구 경기에서 새로운 기록이 나왔다. "
                "선수들은 이번 경기에서 좋은 활약을 보여주었다."
            ),
            "source_url": "https://example.com/sports",
        },

        {
            "title": "프로야구 선수 이적",
            "content": (
                "프로야구 구단이 새로운 선수를 영입했다. "
                "다음 시즌을 위한 선수단 구성이 진행되고 있다."
            ),
            "source_url": "https://example.com/baseball",
        },

        {
            "title": "인공지능 기술 발전",
            "content": (
                "인공지능 기술이 다양한 산업에 활용되고 있다. "
                "AI 서비스 시장도 빠르게 성장하고 있다."
            ),
            "source_url": "https://example.com/ai",
        },
    ]

    print("\nReranker 실행 중...")

    results = rerank_documents(
        query=query,
        documents=test_documents,
        top_k=3
    )

    print()
    print("=" * 60)
    print("Reranker 결과")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print("-" * 60)
        print(f"[{index}]")
        print(f"제목: {result.get('title', '')}")
        print(
            f"Reranker 점수: "
            f"{result.get('reranker_score', 0):.4f}"
        )
        print(
            f"URL: "
            f"{result.get('source_url', '')}"
        )

    print()
    print("=" * 60)
    print("Reranker 테스트 완료")
    print("=" * 60)