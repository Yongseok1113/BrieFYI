"""
RAG 검색 성능 평가 - 최종 통합 리포트

지금까지 따로 돌렸던 두 종류의 평가를 하나로 합친다.

    파트 A: 정답을 미리 아는 가짜 기사 10개 질문 테스트
            (필터 없이 vs 권장 검색, 소수점 점수)

    파트 B: DB에 이미 저장된 "진짜 SBS 기사"들로
            self-retrieval MRR 테스트
            (RSS를 새로 안 불러오고, 이미 저장된 진짜 기사들의
             제목으로 검색해서 자기 자신을 찾아내는지 확인)
"""

import psycopg2

from rag.retriever import retrieve_documents
from rag.eval_dataset import EVAL_QUERIES, seed_eval_data
from rag.db_config import DB_CONFIG


# ==========================================
# 공용 함수
# ==========================================

def reciprocal_rank_score(results: list, key: str, expected_value: str) -> float:
    """
    검색 결과 리스트 안에서 정답을 찾아 Reciprocal Rank(1/순위) 점수로 변환한다.

    key: 정답을 비교할 기준 필드 이름.
         가짜 기사 테스트는 "title"로 비교하고,
         진짜 SBS 기사 테스트는 "source_url"로 비교한다.
         (진짜 기사는 같은 기사가 chunk 여러 개로 쪼개져 있어서
          제목만으로 비교하면 완전히 똑같은 제목이 여러 줄 나와
          헷갈릴 수 있어, URL로 비교하는 게 더 정확하다)
    """

    values = [r.get(key) for r in results]

    if expected_value in values:
        rank = values.index(expected_value) + 1
        return 1 / rank

    return 0.0


def count_total_documents() -> int:

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM rag_documents")
        return cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()


def count_eval_documents() -> int:

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT COUNT(*) FROM rag_documents "
            "WHERE source_url LIKE 'https://example.com/eval/%'"
        )
        return cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()


def get_existing_sbs_articles() -> list:
    """
    RSS를 새로 안 불러오고, DB에 이미 저장돼 있는
    진짜 SBS 기사들의 (제목, source_url) 목록을 가져온다.

    DISTINCT를 쓰는 이유:
        기사 1개가 chunk 여러 개로 쪼개져 저장돼 있으므로,
        같은 source_url이 여러 줄 나오는 걸 하나로 합친다.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT DISTINCT title, source_url
            FROM rag_documents
            WHERE source_url LIKE 'https://news.sbs.co.kr%'
            """
        )

        return [
            {"title": row[0], "source_url": row[1]}
            for row in cur.fetchall()
        ]

    finally:
        cur.close()
        conn.close()


def ensure_eval_data_seeded():

    if count_eval_documents() == 0:
        print("평가용 가짜 기사 없음 -> 새로 저장 중...")
        seed_eval_data()
    else:
        print("평가용 가짜 기사 이미 존재 -> 재사용")


# ==========================================
# 파트 A: 가짜 기사 10개 질문 테스트
# ==========================================

def run_part_a(total_docs: int):

    print("\n" + "-" * 70)
    print("파트 A: 정답을 아는 테스트 질문 10개")
    print("-" * 70)

    rows = []

    for item in EVAL_QUERIES:

        query = item["query"]
        expected_title = item["expected_title"]
        filters = item["filters"]

        without_results = retrieve_documents(query=query, top_k=total_docs)
        without_score = reciprocal_rank_score(without_results, "title", expected_title)

        recommended_results = retrieve_documents(query=query, top_k=5, **filters)
        recommended_score = reciprocal_rank_score(recommended_results, "title", expected_title)

        rows.append(
            {
                "query": query,
                "without_score": without_score,
                "recommended_score": recommended_score,
            }
        )

        print(
            f"  {query} -> 필터 없이: {without_score:.3f}, "
            f"권장 검색: {recommended_score:.3f}"
        )

    return rows


# ==========================================
# 파트 B: 진짜 SBS 기사 self-retrieval 테스트
# ==========================================

def run_part_b():

    print("\n" + "-" * 70)
    print("파트 B: 실제 SBS 기사 self-retrieval (DB에 이미 저장된 것 전부)")
    print("-" * 70)

    articles = get_existing_sbs_articles()

    if not articles:
        print("  DB에 저장된 SBS 기사가 없습니다.")
        return []

    rows = []

    for article in articles:

        title = article["title"]
        url = article["source_url"]

        results = retrieve_documents(query=title, top_k=20)

        score = reciprocal_rank_score(results, "source_url", url)

        rows.append({"title": title, "score": score})

        print(f"  {title} -> {score:.3f}")

    return rows


# ==========================================
# 최종 통합 실행
# ==========================================

def run_all():

    ensure_eval_data_seeded()

    total_docs = count_total_documents()

    print("=" * 70)
    print("검색 성능 최종 통합 리포트")
    print("=" * 70)
    print(f"\n전체 검색 대상: {total_docs}개 chunk")

    part_a_rows = run_part_a(total_docs)
    part_b_rows = run_part_b()

    avg_a_without = sum(r["without_score"] for r in part_a_rows) / len(part_a_rows)
    avg_a_recommended = sum(r["recommended_score"] for r in part_a_rows) / len(part_a_rows)
    avg_b = (
        sum(r["score"] for r in part_b_rows) / len(part_b_rows)
        if part_b_rows else 0.0
    )

    print("\n" + "=" * 70)
    print("표")
    print("=" * 70)

    print("\n[파트 A] 정답을 아는 테스트 질문 10개")
    print("검색어 | 필터 없이 점수 | 권장 검색 점수")
    print("---|---|---")
    for row in part_a_rows:
        print(f"{row['query']} | {row['without_score']:.3f} | {row['recommended_score']:.3f}")

    print(f"\n[파트 A 평균] 필터 없이: {avg_a_without:.3f} / 권장 검색: {avg_a_recommended:.3f}")

    print(f"\n[파트 B] 실제 SBS 기사 {len(part_b_rows)}개 self-retrieval")
    print("기사 제목 | 점수")
    print("---|---")
    for row in part_b_rows:
        print(f"{row['title']} | {row['score']:.3f}")

    print(f"\n[파트 B 평균 = MRR] {avg_b:.3f}")

    print("\n" + "=" * 70)
    print("최종 요약 (3개 숫자)")
    print("=" * 70)
    print(f"1. 가짜 기사 테스트 - 필터 없이:  {avg_a_without:.3f}")
    print(f"2. 가짜 기사 테스트 - 권장 검색:  {avg_a_recommended:.3f}")
    print(f"3. 실제 SBS 기사 - Self-Retrieval MRR: {avg_b:.3f}")


if __name__ == "__main__":
    run_all()

    print("\n" + "=" * 70)
    print("리포트 완료")
    print("=" * 70)