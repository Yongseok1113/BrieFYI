"""
RAG 검색 성능 평가 - Before/After 비교

"메타데이터 필터(category/domain/event)를 걸었을 때"와
"안 걸었을 때"를 같은 질문으로 비교해서,
정답 기사가 몇 번째 순위로 나오는지를 나란히 보여준다.

eval_retriever.py의 Recall@5 = 100%라는 숫자 하나만으로는
"애초에 필터가 없어도 다 잘 찾았던 거 아니야?"라는 질문에
답할 수가 없다.
이 파일은 "필터를 껐을 때와 켰을 때 순위가 실제로 어떻게
달라지는지"를 직접 비교해서 보여줌으로써,
메타데이터 필터가 실제로 검색 정확도에 기여하는지를 증명한다.

검색 결과를 관련도 순으로 쭉 나열했을 때,
정답 기사가 몇 번째 자리에 있는지를 뜻한다.
1위가 제일 좋고, 숫자가 클수록(뒤로 밀릴수록) 안 좋은 것이다.
"검색 결과 안에 아예 없음"인 경우는 "찾지 못함"으로 표시한다.
"""

from rag.retriever import retrieve_documents
from rag.eval_dataset import EVAL_QUERIES, seed_eval_data, cleanup_eval_data


def find_rank(result_titles: list, expected_title: str) -> "int | None":
    """
    검색 결과 제목 리스트 안에서 정답(expected_title)이
    몇 번째(1부터 시작) 자리에 있는지 찾는다.

    result_titles.index(expected_title)
        → 리스트 안에서 그 값이 몇 번째(0부터 시작)에 있는지 찾아준다.
          여기에 +1을 해서 "1위, 2위..." 식으로 사람이 읽기 편하게 바꾼다.

    만약 리스트 안에 아예 없으면 .index()가 에러를 내는데,
    그 경우는 "못 찾음"이라는 뜻으로 None을 돌려준다.
    """

    if expected_title in result_titles:
        return result_titles.index(expected_title) + 1

    return None


def compare_with_without_filter():
    """
    filters가 있는 질문들만 골라서,
    필터 없이 검색한 결과와 필터 적용해서 검색한 결과를
    나란히 비교한다.
    """

    print("=" * 70)
    print("검색 성능 비교: 필터 없음 vs 메타데이터 필터 적용")
    print("=" * 70)

    # --------------------------------------
    # filters가 있는 질문만 골라낸다.
    # (filters가 {}인 질문은 비교할 대상이 없으므로 건너뜀)
    # --------------------------------------
    filtered_queries = [
        item for item in EVAL_QUERIES if item["filters"]
    ]

    comparison_rows = []

    for item in filtered_queries:

        query = item["query"]
        expected_title = item["expected_title"]
        filters = item["filters"]

        # --------------------------------------
        # 1. 필터 없이 검색 (전체 12개 기사 중에서 찾아야 하므로
        #    top_k를 넉넉하게 12로 잡아서, 순위가 5위 밖이어도
        #    몇 위인지까지 확인할 수 있게 한다)
        # --------------------------------------
        without_filter_results = retrieve_documents(
            query=query,
            top_k=12,
        )
        without_titles = [r["title"] for r in without_filter_results]
        rank_without = find_rank(without_titles, expected_title)

        # --------------------------------------
        # 2. 필터 적용해서 검색
        # --------------------------------------
        with_filter_results = retrieve_documents(
            query=query,
            top_k=5,
            **filters,
        )
        with_titles = [r["title"] for r in with_filter_results]
        rank_with = find_rank(with_titles, expected_title)

        comparison_rows.append(
            {
                "query": query,
                "filters": filters,
                "rank_without": rank_without,
                "rank_with": rank_with,
            }
        )

        print(f"\n검색어: {query}")
        print(f"  필터: {filters}")
        print(f"  필터 없이 검색 → 정답 순위: {rank_without if rank_without else '못 찾음'}위")
        print(f"  필터 적용 검색 → 정답 순위: {rank_with if rank_with else '못 찾음'}위")

    # --------------------------------------
    # 마크다운 표
    # --------------------------------------

    print("\n" + "=" * 70)
    print("요약 표")
    print("=" * 70)
    print()
    print("검색어 | 필터 없이 | 필터 적용")
    print("---|---|---")

    for row in comparison_rows:

        without_display = (
            f"{row['rank_without']}위" if row['rank_without'] else "못 찾음"
        )
        with_display = (
            f"{row['rank_with']}위" if row['rank_with'] else "못 찾음"
        )

        print(f"{row['query']} | {without_display} | {with_display}")

    # --------------------------------------
    # 평균 순위 계산
    #
    # None(못 찾음)은 평균 계산에서 제외하고,
    # 실제로 순위가 나온 것들만으로 평균을 낸다.
    # --------------------------------------

    ranks_without = [r["rank_without"] for r in comparison_rows if r["rank_without"]]
    ranks_with = [r["rank_with"] for r in comparison_rows if r["rank_with"]]

    avg_without = sum(ranks_without) / len(ranks_without) if ranks_without else 0
    avg_with = sum(ranks_with) / len(ranks_with) if ranks_with else 0

    print()
    print(f"평균 순위 (필터 없이): {avg_without:.1f}위")
    print(f"평균 순위 (필터 적용): {avg_with:.1f}위")

    return comparison_rows


# ==========================================
# 테스트
# ==========================================

if __name__ == "__main__":

    # 평가용 가짜 기사가 DB에 없을 수도 있으니
    # (지난번에 지웠을 수 있으므로) 여기서 다시 심어준다.
    print("평가용 데이터 준비 중...")
    seed_eval_data()

    compare_with_without_filter()

    print("\n" + "=" * 70)
    print("비교 완료")
    print("=" * 70)
    print("\n평가용 가짜 데이터를 지우려면 아래 명령을 실행하세요:")
    print('  python -c "from rag.eval_dataset import cleanup_eval_data; cleanup_eval_data()"')