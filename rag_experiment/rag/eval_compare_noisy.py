"""
RAG 검색 성능 평가 - 노이즈(SBS 정치 기사) 섞어서 비교

정답을 아는 가짜 기사 12개 + 이미 DB에 있는 SBS 정치 기사들(노이즈)을
같이 검색 대상으로 두고, 필터 없이 vs 필터 적용 성능을 비교한다.

eval_compare.py는 가짜 기사 12개끼리만 비교해서
주제가 너무 뚜렷하게 갈리다 보니 필터가 있든 없든 항상 1위로 나왔다.
실제 서비스에서는 기사가 더 많이 쌓이는데,
그 안에서도 필터가 정답을 정확히 찾아주는지가 궁금한 부분이다.
그래서 지금 DB에 이미 있는 SBS 정치 기사(전부 category="기타")를 두고,
그 사이에서도 검색이 잘 되는지 확인한다.

category/domain/event 필터를 걸면 "기타"인 기사들은
자동으로 검색 대상에서 제외된다.
반면 필터 없이 검색하면 이 노이즈 기사들도 전부 후보에 끼어들어
벡터 유사도만으로 순위를 다퉈야 한다.
→ "필터가 노이즈를 걸러내는 효과"를 그대로 보여줄 수 있다.
"""

import psycopg2

from rag.retriever import retrieve_documents
from rag.eval_dataset import EVAL_QUERIES, seed_eval_data, cleanup_eval_data
from rag.db_config import DB_CONFIG


def count_documents(source_url_prefix: str = None) -> int:
    """
    현재 rag_documents 테이블에 몇 개의 chunk가 있는지 센다.

    source_url_prefix가 주어지면, 그 접두어로 시작하는 것만 센다.
    (예: "https://example.com/eval/"만 세면 가짜 평가 기사 개수만 나옴)
    안 주면 테이블 전체 개수를 센다.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        if source_url_prefix:
            cur.execute(
                "SELECT COUNT(*) FROM rag_documents WHERE source_url LIKE %s",
                (f"{source_url_prefix}%",),
            )
        else:
            cur.execute("SELECT COUNT(*) FROM rag_documents")

        return cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()


def ensure_eval_data_seeded():
    """
    가짜 평가 기사 12개가 이미 DB에 있으면 다시 안 넣고,
    없으면 새로 넣는다.

    (이미 있는데 또 넣으면 중복으로 쌓이는 걸 방지하기 위한 안전장치)
    """

    existing_count = count_documents("https://example.com/eval/")

    if existing_count > 0:
        print(f"평가용 가짜 기사가 이미 {existing_count}개 있음 → 다시 안 넣음")
    else:
        print("평가용 가짜 기사 없음 → 새로 저장 중...")
        seed_eval_data()


def find_rank(result_titles: list, expected_title: str) -> "int | None":
    """
    검색 결과 제목 리스트 안에서 정답이 몇 번째 자리에 있는지 찾는다.
    (eval_compare.py와 동일한 함수)
    """

    if expected_title in result_titles:
        return result_titles.index(expected_title) + 1

    return None


def compare_with_noise():
    """
    노이즈(SBS 정치 기사)가 섞인 상태로
    필터 없이 vs 필터 적용 검색 순위를 비교한다.
    """

    ensure_eval_data_seeded()

    # --------------------------------------
    # 지금 검색 대상 전체가 몇 개인지 먼저 보여준다.
    # --------------------------------------
    total_docs = count_documents()
    eval_docs = count_documents("https://example.com/eval/")
    noise_docs = total_docs - eval_docs

    print("=" * 70)
    print("검색 성능 비교 (노이즈 포함)")
    print("=" * 70)
    print(f"\n전체 검색 대상: {total_docs}개 chunk")
    print(f"  - 정답 아는 평가용 기사: {eval_docs}개")
    print(f"  - 노이즈(SBS 정치 기사 등): {noise_docs}개")

    filtered_queries = [
        item for item in EVAL_QUERIES if item["filters"]
    ]

    comparison_rows = []

    for item in filtered_queries:

        query = item["query"]
        expected_title = item["expected_title"]
        filters = item["filters"]

        # --------------------------------------
        # 필터 없이 검색할 때는 노이즈까지 전부 후보에 들어오므로,
        # top_k를 "전체 문서 개수"만큼 넉넉하게 잡아서
        # 정답이 몇 등으로 밀렸는지 끝까지 확인할 수 있게 한다.
        # --------------------------------------
        without_filter_results = retrieve_documents(
            query=query,
            top_k=total_docs,
        )
        without_titles = [r["title"] for r in without_filter_results]
        rank_without = find_rank(without_titles, expected_title)

        # --------------------------------------
        # 필터 적용 검색은 그대로 top_k=5
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

        without_display = f"{rank_without}위" if rank_without else "못 찾음"
        with_display = f"{rank_with}위" if rank_with else "못 찾음"

        print(f"\n검색어: {query}")
        print(f"  필터: {filters}")
        print(f"  필터 없이 (노이즈 {noise_docs}개 포함 중 검색) → 정답 순위: {without_display}")
        print(f"  필터 적용 (노이즈 자동 제외) → 정답 순위: {with_display}")

    # --------------------------------------
    # 마크다운 표
    # --------------------------------------

    print("\n" + "=" * 70)
    print(f"요약 표 (전체 {total_docs}개 chunk 중 검색)")
    print("=" * 70)
    print()
    print("검색어 | 필터 없이 (노이즈 포함) | 필터 적용")
    print("---|---|---")

    for row in comparison_rows:

        without_display = (
            f"{row['rank_without']}위" if row['rank_without'] else "못 찾음"
        )
        with_display = (
            f"{row['rank_with']}위" if row['rank_with'] else "못 찾음"
        )

        print(f"{row['query']} | {without_display} | {with_display}")

    return comparison_rows


if __name__ == "__main__":

    compare_with_noise()

    print("\n" + "=" * 70)
    print("비교 완료")
    print("=" * 70)
    print("\n평가용 가짜 데이터만 지우려면 (SBS 노이즈는 안 지워짐):")
    print('  python -c "from rag.eval_dataset import cleanup_eval_data; cleanup_eval_data()"')