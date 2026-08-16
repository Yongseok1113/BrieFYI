"""
RAG 검색 성능 평가 - Recall@5 측정

eval_dataset.py에 정의된 10개 테스트 질문을 실제로 검색해보고,
"정답 기사가 상위 5개 안에 포함됐는가"를 세어서
Recall@5라는 하나의 점수로 정리한다.

Recall@5이란?
"정답이 있는 질문 중에서, 검색 결과 상위 5개 안에
그 정답이 실제로 포함된 비율"을 뜻한다.

예: 질문 10개 중 8개는 정답이 상위 5개 안에 있었고,
    2개는 없었다면 → Recall@5 = 8/10 = 80%

이 지표가 중요한 이유?
Reranker가 아무리 좋아도,
Retriever가 애초에 정답 후보를 안 가져오면
Reranker는 그 정답을 재정렬조차 할 수 없다.
그래서 "Retriever가 정답을 후보 안에 넣어주는지"를
먼저 확인하는 게 Recall@5의 역할이다.
"""

from rag.retriever import retrieve_documents
from rag.eval_dataset import EVAL_QUERIES


def evaluate_recall_at_k(k: int = 5):
    """
    EVAL_QUERIES에 있는 질문들을 하나씩 검색해보고
    Recall@k를 계산한다.

    Args:
        k: "상위 몇 개까지 볼 것인가" (기본값 5 → Recall@5)
    """

    print("=" * 60)
    print(f"검색 성능 평가 (Recall@{k})")
    print("=" * 60)

    # --------------------------------------
    # 맞춘 질문 개수를 세는 카운터.
    # 하나씩 검색해보면서, 정답을 찾으면 +1씩 늘어난다.
    # --------------------------------------
    hit_count = 0

    total_count = len(EVAL_QUERIES)

    # 질문 하나하나에 대한 상세 결과를 나중에 표로 다시 보여주기 위해
    # 리스트에 차곡차곡 쌓아둔다.
    detail_results = []

    for index, eval_item in enumerate(EVAL_QUERIES, start=1):

        query = eval_item["query"]
        expected_title = eval_item["expected_title"]
        filters = eval_item["filters"]

        print(f"\n[{index}/{total_count}] 검색어: {query}")

        if filters:
            print(f"  필터: {filters}")

        # --------------------------------------
        # filters 딕셔너리를 retrieve_documents()의
        # 개별 파라미터(category=, domain=...)로 풀어서 넘긴다.
        #
        # **filters 문법:
        #   filters가 {"domain": "스타트업"} 이라면
        #   retrieve_documents(query=query, top_k=k, domain="스타트업")
        #   과 완전히 똑같은 뜻이 된다.
        #   (딕셔너리를 "키=값" 형태의 파라미터들로 자동으로 풀어줌)
        # --------------------------------------

        results = retrieve_documents(
            query=query,
            top_k=k,
            **filters,
        )

        # 검색 결과로 나온 기사 제목들만 리스트로 뽑아둔다.
        result_titles = [r["title"] for r in results]

        # --------------------------------------
        # 정답(expected_title)이 검색 결과 제목 리스트 안에
        # 있는지 확인한다. (in 연산자: "포함돼 있는가?")
        # --------------------------------------

        is_hit = expected_title in result_titles

        if is_hit:
            hit_count += 1
            print(f"  결과: 성공 (정답이 상위 {k}개 안에 있음)")
        else:
            print(f"  결과: 실패 (정답을 못 찾음)")
            print(f"  기대한 정답: {expected_title}")
            print(f"  실제 검색된 제목들:")
            for title in result_titles:
                print(f"    - {title}")

        detail_results.append(
            {
                "query": query,
                "expected_title": expected_title,
                "is_hit": is_hit,
                "result_titles": result_titles,
            }
        )

    # --------------------------------------
    # 최종 Recall@k 계산 및 출력
    #
    # hit_count / total_count
    #   → 정답 맞춘 개수를 전체 질문 개수로 나눈 비율.
    # --------------------------------------

    recall_at_k = hit_count / total_count

    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)

    print(f"\n전체 질문: {total_count}개")
    print(f"정답 찾음: {hit_count}개")

    # :.1% 는 "소수를 퍼센트 형식으로, 소수점 1자리까지" 표시하는 문법.
    # 예: 0.8 → "80.0%"
    print(f"Recall@{k}: {recall_at_k:.1%}")

    return {
        "recall_at_k": recall_at_k,
        "hit_count": hit_count,
        "total_count": total_count,
        "details": detail_results,
    }


# ==========================================
# 테스트
# ==========================================

if __name__ == "__main__":

    evaluate_recall_at_k(k=5)

    print("\n" + "=" * 60)
    print("평가 완료")
    print("=" * 60)
    print("\n평가용 가짜 데이터를 지우려면 아래 명령을 실행하세요:")
    print('  python -c "from rag.eval_dataset import cleanup_eval_data; cleanup_eval_data()"')