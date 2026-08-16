"""
RAG 에이전트용 검색 툴

에이전트(LLM)가 "엔비디아 AI 투자 뉴스 찾아줘" 같은 요청을 받았을 때
호출할 수 있는 함수(search_news)를 만든다.

1. retriever.py의 retrieve_documents()로 후보 기사를 넉넉히 가져온다
   (category/domain/entity/event 필터도 그대로 받아서 넘긴다)
2. reranker.py의 rerank_documents()로 그 후보들을 재정렬한다
3. 결과를 "JSON으로 그대로 내보내도 문제없는 형태"로 정리한다

3번이 필요한 이유
DB에서 가져온 데이터 중에는 파이썬 datetime 객체(published_at)처럼
JSON으로 바로 변환하면 에러가 나는 타입이 섞여있다.
그래서 에이전트에게 결과를 넘기기 전에
전부 문자열/숫자/리스트 같은 "JSON이 이해할 수 있는 형태"로
미리 한 번 정리해주는 것이다.
"""

from typing import List, Dict, Any, Optional

# json은 테스트 코드에서 "진짜 JSON으로 잘 변환되는지"
# 확인해보는 용도로만 쓴다.
import json

from rag.retriever import retrieve_documents
from rag.reranker import rerank_documents
from rag.embed import unload_model

# ==========================================
# 1. 에이전트용 검색 함수
# ==========================================

def search_news(
    query: str,
    category: Optional[str] = None,
    domain: Optional[str] = None,
    entity: Optional[str] = None,
    event: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    에이전트가 호출하는 단일 진입점(entry point) 함수.

    Args:
        query: 자연어 검색어 (예: "엔비디아 AI 투자 뉴스")
        category: 대분류 필터 (예: "기술"), 없으면 None
        domain: 도메인 필터 (예: "AI"), 없으면 None
        entity: 엔티티 필터 (예: "NVIDIA"), 없으면 None
        event: 이벤트 필터 (예: "투자"), 없으면 None
        top_k: 최종적으로 몇 개 결과를 돌려줄지

    Returns:
        JSON으로 바로 변환 가능한 딕셔너리 리스트
    """

    # --------------------------------------
    # 1단계: Retriever로 후보를 충분히 가져온다
    #
    # top_k * 3 인 이유:
    #   Reranker가 후보가 5개보다는 많아야 그중에서 진짜 좋은 걸 고를 수 있음
    #   (Reranker는 "고르는" 역할)
    # --------------------------------------

    candidates = retrieve_documents(
        query=query,
        top_k=top_k * 3,
        category=category,
        domain=domain,
        entity=entity,
        event=event,
    )

    if not candidates:
        return []

    # --------------------------------------
    # 리랭커 모델을 로딩하기 전에
    # 임베딩 모델을 먼저 메모리에서 내려놓는다.
    # (두 모델을 동시에 메모리에 올려서 죽는 문제 방지)
    # --------------------------------------
    unload_model()

    # --------------------------------------
    # 2단계: Reranker로 진짜 관련 있는 것만 추려낸다
    # --------------------------------------

    reranked = rerank_documents(
        query=query,
        documents=candidates,
        top_k=top_k,
    )

    # --------------------------------------
    # 3단계: JSON으로 안전하게 내보낼 수 있는 형태로 정리
    #
    # doc.get("키", 기본값)
    #   → 딕셔너리에서 "키"에 해당하는 값을 꺼내되,
    #     그 키가 없으면 에러 내지 않고 기본값을 대신 쓴다.
    # --------------------------------------

    results = []

    for doc in reranked:

        results.append(
            {
                "title": doc.get("title", ""),

                # content가 너무 길면 에이전트에게 넘기는 데이터量이
                # 불필요하게 커지므로, 앞부분 500자만 잘라서 준다.
                "content": (doc.get("content") or "")[:500],

                "source_url": doc.get("source_url", ""),

                # published_at은 DB에서 datetime 객체로 넘어오는데,
                # datetime은 json.dumps()가 그대로 처리 못 하는 타입이다.
                # str()로 감싸서 "2026-08-16 10:00:00" 같은
                # 평범한 문자열로 바꿔준다.
                # 값이 아예 없는 경우 (None)
                "published_at": (
                    str(doc["published_at"])
                    if doc.get("published_at")
                    else None
                ),

                "category": doc.get("category"),
                "domain": doc.get("domain"),
                "entities": doc.get("entities") or [],
                "event": doc.get("event"),

                # reranker_score는 float라서 JSON 변환에 문제없지만,
                # round()로 소수점 4자리까지만 남겨서 보기 편하게 정리.
                "relevance_score": round(
                    doc.get("reranker_score", 0),
                    4
                ),
            }
        )

    return results


# ==========================================
# 2. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("Agent Search 테스트")
    print("=" * 60)

    # 지금 DB에 정치 기사만 있으니, 그 안에서 실제로
    # 잘 검색되는지 확인하는 용도로 테스트한다.
    query = "김민석 관련 뉴스"

    results = search_news(
        query=query,
        top_k=3,
    )

    print(f"\n검색어: {query}")
    print(f"결과: {len(results)}개\n")

    # --------------------------------------
    # 진짜로 JSON 직렬화가 되는지 직접 확인해본다.
    #
    # json.dumps()가 에러 없이 실행되면
    # = 이 결과를 그대로 에이전트에게 JSON으로 넘겨도
    #   안전하다는 뜻이다.
    # --------------------------------------

    json_output = json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    )

    print(json_output)

    print("\n" + "=" * 60)
    print("Agent Search 테스트 완료 (JSON 직렬화 확인됨)")
    print("=" * 60)