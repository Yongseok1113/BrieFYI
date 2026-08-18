"""
RAG 검색 성능 평가 - 최종 통합 리포트

파트 A: 정답을 아는 가짜 기사 10개 질문 테스트 (필터 없이 vs 권장 검색)
파트 B: DB에 있는 진짜 SBS 기사 전체로 self-retrieval 평가
        (본문이 너무 짧은 크롤링 실패 기사는 제외, 점수 낮은 순 정렬)
"""

import psycopg2

from rag.retriever import retrieve_documents
from rag.eval_dataset import EVAL_QUERIES, seed_eval_data
from rag.db_config import DB_CONFIG


# ==========================================
# 공용 함수
# ==========================================

def reciprocal_rank_score(results: list, key: str, expected_value: str) -> float:

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
            "WHERE source_url LIKE 'https://example.com/eval/%%'"
        )
        return cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()


def get_all_sbs_articles(min_content_length=100) -> list:
    """
    본문이 min_content_length자 이상인 진짜 SBS 기사만 가져온다.
    (영상/카드뉴스형 콘텐츠 등 크롤링 실패로 본문이 너무 짧은 것은 제외)
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT title, source_url, SUM(LENGTH(content)) as total_len
            FROM rag_documents
            WHERE source_url LIKE 'https://news.sbs.co.kr%%'
            GROUP BY title, source_url
            HAVING SUM(LENGTH(content)) >= %s
            """,
            (min_content_length,)
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

        rows.append({
            "query": query,
            "without_score": without_score,
            "recommended_score": recommended_score,
        })

        print(f"  {query} -> 필터 없이: {without_score:.3f}, 권장 검색: {recommended_score:.3f}")

    return rows


# ==========================================
# 파트 B: 진짜 SBS 기사 self-retrieval (대규모, 정렬된 표)
# ==========================================

def run_part_b(min_content_length=100):

    print("\n" + "-" * 70)
    print("파트 B: 실제 SBS 기사 self-retrieval (본문 짧은 기사 제외)")
    print("-" * 70)

    articles = get_all_sbs_articles(min_content_length=min_content_length)

    total = len(articles)

    if total == 0:
        print("  평가할 SBS 기사가 없습니다.")
        return [], total

    rows = []

    for index, article in enumerate(articles, start=1):

        title = article["title"]
        url = article["source_url"]

        results = retrieve_documents(query=title, top_k=20)

        values = [r.get("source_url") for r in results]
        rank = values.index(url) + 1 if url in values else None

        score = (1 / rank) if rank else 0.0

        rows.append({"title": title, "rank": rank, "score": score})

        # 원본(eval_large_scale.py)처럼 기사마다 즉시 결과 출력
        rank_display = f"{rank}위" if rank else "20위 밖"
        print(f"[{index}/{total}] {title} -> {rank_display} (점수: {score:.3f})")

    rows.sort(key=lambda r: r["score"])

    return rows, total


# ==========================================
# 최종 통합 실행
# ==========================================

def run_all(min_content_length=100):

    ensure_eval_data_seeded()

    total_docs = count_total_documents()

    print("=" * 70)
    print("검색 성능 최종 통합 리포트")
    print("=" * 70)
    print(f"\n전체 DB 문서(chunk) 수: {total_docs}개")

    part_a_rows = run_part_a(total_docs)
    part_b_rows, part_b_total = run_part_b(min_content_length=min_content_length)

    avg_a_without = sum(r["without_score"] for r in part_a_rows) / len(part_a_rows)
    avg_a_recommended = sum(r["recommended_score"] for r in part_a_rows) / len(part_a_rows)

    if part_b_total > 0:
        hit_at_5 = sum(1 for r in part_b_rows if r["rank"] and r["rank"] <= 5)
        hit_at_10 = sum(1 for r in part_b_rows if r["rank"] and r["rank"] <= 10)
        recall_at_5 = hit_at_5 / part_b_total
        recall_at_10 = hit_at_10 / part_b_total
        mrr_b = sum(r["score"] for r in part_b_rows) / part_b_total
    else:
        recall_at_5 = recall_at_10 = mrr_b = 0.0

    print("\n" + "=" * 70)
    print("팀 공유용 최종 표")
    print("=" * 70)

    print("\n[파트 A] 정답을 아는 테스트 질문 10개")
    print("검색어 | 필터 없이 점수 | 권장 검색 점수")
    print("---|---|---")
    for row in part_a_rows:
        print(f"{row['query']} | {row['without_score']:.3f} | {row['recommended_score']:.3f}")
    print(f"\n[파트 A 평균] 필터 없이: {avg_a_without:.3f} / 권장 검색: {avg_a_recommended:.3f}")

    print(f"\n[파트 B] 실제 SBS 기사 {part_b_total}개 self-retrieval (점수 낮은 순)")
    print("기사 제목 | 순위 | 점수")
    print("---|---|---")
    for row in part_b_rows:
        rank_display = f"{row['rank']}위" if row["rank"] else "못 찾음"
        print(f"{row['title']} | {rank_display} | {row['score']:.3f}")

    print("\n" + "=" * 70)
    print("최종 요약")
    print("=" * 70)
    print(f"1. [파트 A] 가짜 기사 - 필터 없이:  {avg_a_without:.3f}")
    print(f"2. [파트 A] 가짜 기사 - 권장 검색:  {avg_a_recommended:.3f}")
    print(f"3. [파트 B] 실제 SBS 기사 {part_b_total}개 - Recall@5:  {recall_at_5:.3f}")
    print(f"4. [파트 B] 실제 SBS 기사 {part_b_total}개 - Recall@10: {recall_at_10:.3f}")
    print(f"5. [파트 B] 실제 SBS 기사 {part_b_total}개 - MRR:       {mrr_b:.3f}")


if __name__ == "__main__":
    run_all()

    print("\n" + "=" * 70)
    print("리포트 완료")
    print("=" * 70)