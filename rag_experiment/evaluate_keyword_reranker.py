"""
키워드별 검색 성능 평가 (리랭커 적용 전/후 비교)

  1단계: 임베딩 모델로 7개 키워드 전부 pgvector 검색까지 끝내고
         결과(candidates)를 리스트에 저장해둔다.
  2단계: 임베딩 모델을 메모리에서 내리고, 그 다음 리랭커를 로딩해서
         저장해둔 결과들을 순서대로 리랭킹한다.

"""

import gc
import json
import os

# OpenMP 중복 로딩 충돌 방지 (Windows에서 torch 계열 라이브러리 여러 개
# 같이 쓸 때 강제 종료되는 문제를 막아준다)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import psycopg2

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PostgreSQL 연결
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password"
)

cur = conn.cursor()


# ==========================================
# 2. Gold 데이터 불러오기
# ==========================================

with open(
    "gold_keyword_candidates.json",
    "r",
    encoding="utf-8"
) as f:

    gold_data = json.load(f)

print(f"평가 키워드 수: {len(gold_data)}")


# ==========================================
# 3단계: 임베딩 모델로 pgvector 검색까지만 진행
# ==========================================

print("\nEmbedding 모델 로딩 중...")

embedding_model = SentenceTransformer("BAAI/bge-m3")

# 키워드별로 "질문", "정답 목록", "필터 전 후보(candidates)"를
# 여기에 다 저장해둔다. 이 시점에는 아직 리랭킹은 안 한다.
prepared_items = []

for item in gold_data:

    keyword = item["keyword"]
    relevant_titles = set(item["relevant_titles"])

    query = f"{keyword} 관련 최신 뉴스"

    query_embedding = embedding_model.encode(query).tolist()

    cur.execute(
        """
        SELECT
            title,
            content,
            source_url,
            embedding <=> %s::vector AS distance
        FROM rag_documents
        ORDER BY embedding <=> %s::vector
        LIMIT 15
        """,
        (query_embedding, query_embedding),
    )

    candidates = cur.fetchall()

    prepared_items.append(
        {
            "keyword": keyword,
            "query": query,
            "relevant_titles": relevant_titles,
            "candidates": candidates,
        }
    )

    print(f"  임베딩+검색 완료: {keyword} ({len(candidates)}개 후보)")

# --------------------------------------
# 임베딩 모델을 메모리에서 완전히 내린다.
# 이제부터는 embedding_model을 다시 쓰지 않는다.
# --------------------------------------

print("\nEmbedding 모델 메모리에서 해제 중...")

del embedding_model
gc.collect()

print("해제 완료")


# ==========================================
# 4단계: 리랭커 로딩 후, 저장해둔 후보들 리랭킹
# ==========================================

print("\nReranker 모델 로딩 중...")

from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=False
)


# ==========================================
# 5. 평가 결과 저장용 변수
# ==========================================

all_results = []

recall_1_before, recall_3_before, recall_5_before = [], [], []
recall_1_after, recall_3_after, recall_5_after = [], [], []
mrr_before, mrr_after = [], []


# ==========================================
# 6. 키워드별 리랭킹 + 평가
# ==========================================

for prepared in prepared_items:

    keyword = prepared["keyword"]
    query = prepared["query"]
    relevant_titles = prepared["relevant_titles"]
    candidates = prepared["candidates"]

    print("\n" + "=" * 80)
    print(f"키워드: {keyword}")
    print("=" * 80)

    if not candidates:
        print("후보 기사가 없습니다.")
        continue

    before_results = candidates

    print("\n[리랭커 적용 전 - pgvector 순위]")

    for rank, result in enumerate(before_results, start=1):
        title = result[0]
        distance = result[3]
        mark = "<- 정답" if title in relevant_titles else ""
        print(f"{rank:2d}위 | distance={distance:.4f} | {title} {mark}")

    # --------------------------------------
    # BGE Reranker 실행
    # --------------------------------------

    rerank_pairs = [[query, result[1]] for result in candidates]

    rerank_scores = reranker.compute_score(rerank_pairs, normalize=True)

    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    after_results = []

    for result, score in zip(candidates, rerank_scores):
        after_results.append(
            {
                "title": result[0],
                "content": result[1],
                "source_url": result[2],
                "distance": result[3],
                "rerank_score": score,
            }
        )

    after_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    print("\n[리랭커 적용 후 - BGE Reranker 순위]")

    for rank, result in enumerate(after_results, start=1):
        title = result["title"]
        score = result["rerank_score"]
        mark = "<- 정답" if title in relevant_titles else ""
        print(f"{rank:2d}위 | score={score:.4f} | {title} {mark}")

    # --------------------------------------
    # 정답 기사 순위 계산
    # --------------------------------------

    before_ranks = [
        rank for rank, result in enumerate(before_results, start=1)
        if result[0] in relevant_titles
    ]

    after_ranks = [
        rank for rank, result in enumerate(after_results, start=1)
        if result["title"] in relevant_titles
    ]

    before_recall_1 = int(any(r <= 1 for r in before_ranks))
    before_recall_3 = int(any(r <= 3 for r in before_ranks))
    before_recall_5 = int(any(r <= 5 for r in before_ranks))

    after_recall_1 = int(any(r <= 1 for r in after_ranks))
    after_recall_3 = int(any(r <= 3 for r in after_ranks))
    after_recall_5 = int(any(r <= 5 for r in after_ranks))

    recall_1_before.append(before_recall_1)
    recall_3_before.append(before_recall_3)
    recall_5_before.append(before_recall_5)

    recall_1_after.append(after_recall_1)
    recall_3_after.append(after_recall_3)
    recall_5_after.append(after_recall_5)

    before_mrr = (1 / min(before_ranks)) if before_ranks else 0
    after_mrr = (1 / min(after_ranks)) if after_ranks else 0

    mrr_before.append(before_mrr)
    mrr_after.append(after_mrr)

    print("\n[평가 결과]")
    print(f"정답 기사 수: {len(relevant_titles)}")
    print(f"리랭커 적용 전 정답 순위: {before_ranks if before_ranks else '없음'}")
    print(f"리랭커 적용 후 정답 순위: {after_ranks if after_ranks else '없음'}")
    print(f"Recall@1: {before_recall_1} -> {after_recall_1}")
    print(f"Recall@3: {before_recall_3} -> {after_recall_3}")
    print(f"Recall@5: {before_recall_5} -> {after_recall_5}")
    print(f"MRR: {before_mrr:.4f} -> {after_mrr:.4f}")

    all_results.append(
        {
            "keyword": keyword,
            "before_ranks": before_ranks,
            "after_ranks": after_ranks,
            "recall@1_before": before_recall_1,
            "recall@3_before": before_recall_3,
            "recall@5_before": before_recall_5,
            "recall@1_after": after_recall_1,
            "recall@3_after": after_recall_3,
            "recall@5_after": after_recall_5,
            "mrr_before": before_mrr,
            "mrr_after": after_mrr,
        }
    )


# ==========================================
# 7. 전체 평균 계산
# ==========================================

def average(values):
    return sum(values) / len(values) if values else 0


avg_recall_1_before = average(recall_1_before)
avg_recall_3_before = average(recall_3_before)
avg_recall_5_before = average(recall_5_before)

avg_recall_1_after = average(recall_1_after)
avg_recall_3_after = average(recall_3_after)
avg_recall_5_after = average(recall_5_after)

avg_mrr_before = average(mrr_before)
avg_mrr_after = average(mrr_after)


# ==========================================
# 8. 최종 평가 결과 출력
# ==========================================

print("\n\n" + "=" * 80)
print("전체 평가 결과")
print("=" * 80)

print("\n[Recall]")
print(f"Recall@1 : {avg_recall_1_before:.4f} -> {avg_recall_1_after:.4f}")
print(f"Recall@3 : {avg_recall_3_before:.4f} -> {avg_recall_3_after:.4f}")
print(f"Recall@5 : {avg_recall_5_before:.4f} -> {avg_recall_5_after:.4f}")

print("\n[MRR]")
print(f"MRR       : {avg_mrr_before:.4f} -> {avg_mrr_after:.4f}")

print("\n[개선폭]")
print(f"Recall@1 개선: {avg_recall_1_after - avg_recall_1_before:+.4f}")
print(f"Recall@3 개선: {avg_recall_3_after - avg_recall_3_before:+.4f}")
print(f"Recall@5 개선: {avg_recall_5_after - avg_recall_5_before:+.4f}")
print(f"MRR 개선: {avg_mrr_after - avg_mrr_before:+.4f}")


# ==========================================
# 9. JSON 저장
# ==========================================

evaluation_summary = {
    "num_keywords": len(all_results),
    "recall@1": {"before": avg_recall_1_before, "after": avg_recall_1_after},
    "recall@3": {"before": avg_recall_3_before, "after": avg_recall_3_after},
    "recall@5": {"before": avg_recall_5_before, "after": avg_recall_5_after},
    "mrr": {"before": avg_mrr_before, "after": avg_mrr_after},
    "details": all_results,
}

with open(
    "keyword_reranker_evaluation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)


# ==========================================
# 10. 종료
# ==========================================

cur.close()
conn.close()

print("\n" + "=" * 80)
print("평가 완료")
print("=" * 80)
print("\n결과 파일: keyword_reranker_evaluation.json")