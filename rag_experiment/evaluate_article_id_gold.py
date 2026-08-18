"""
article_id 기반 Gold Set 평가 (STEP 4~8)

     정답을 "article_id"(고유 번호)로 매칭 -> 중복 제목이 있어도
    정확히 그 한 행만 정답으로 잡아서 훨씬 정확함.
"""

import gc
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import psycopg2
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 4. Gold Set
# ==========================================
# 각 쿼리마다 "정답 article_id 목록"을 리스트로 넣게 만들었다.
# (지금은 각 쿼리당 1개씩이지만, 나중에 정답이 여러 개인 쿼리가 생겨도
#  바로 대응 가능하게 리스트 형태로 설계함)
#
# 지금은 예시로 주어진 4개만 들어있다.
# 나머지 6개도 같은 형식으로 아래 리스트에 추가하면 된다:
#   {"field": "분야", "keyword": "키워드", "query": "질문", "relevant_ids": [article_id]}

GOLD_SET = [
    {
        "field": "야구",
        "keyword": "이정후",
        "query": "이정후의 최근 경기 성적과 시즌 타율은?",
        "relevant_ids": [578],
    },
    {
        "field": "삼성",
        "keyword": "삼성전자 반도체 실적",
        "query": "삼성전자 반도체 부문의 최근 실적과 실적 개선 이유는?",
        "relevant_ids": [153],
    },
    {
        "field": "축구",
        "keyword": "호날두",
        "query": "호날두가 은퇴를 시사한 이유와 시점은?",
        "relevant_ids": [572],
    },
    {
        "field": "반도체",
        "keyword": "D램",
        "query": "올해 3분기 PC용 D램 가격은 어떻게 전망돼?",
        "relevant_ids": [348],
    },
    # ↓↓↓ 나머지 6개 쿼리, 준비되면 이 자리에 같은 형식으로 추가 ↓↓↓
]


# ==========================================
# PostgreSQL 연결
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password",
)

cur = conn.cursor()


# ==========================================
# STEP 5. 벡터 검색 Top-10 (임베딩 모델 쓰는 구간)
# ==========================================

print(f"평가 쿼리 수: {len(GOLD_SET)}")
print("\nEmbedding 모델 로딩 중...")

embedding_model = SentenceTransformer("BAAI/bge-m3")

prepared_items = []

for item in GOLD_SET:

    query = item["query"]
    query_embedding = embedding_model.encode(query).tolist()

    cur.execute(
        """
        SELECT
            id,
            title,
            content,
            source_url,
            embedding <=> %s::vector AS distance
        FROM rag_documents
        ORDER BY embedding <=> %s::vector
        LIMIT 10
        """,
        (query_embedding, query_embedding),
    )

    top10 = cur.fetchall()

    prepared_items.append({**item, "top10": top10})

    print(f"  검색 완료: [{item['field']}] {item['query'][:30]}... ({len(top10)}개)")

print("\nEmbedding 모델 메모리에서 해제 중...")
del embedding_model
gc.collect()
print("해제 완료")


# ==========================================
# STEP 6. BGE Reranker로 같은 Top-10 재정렬
# ==========================================

print("\nReranker 모델 로딩 중...")

from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)


# ==========================================
# STEP 7. 쿼리별 전/후 비교
# ==========================================

def find_rank(ordered_ids: list[int], relevant_ids: set[int]):
    """
    ordered_ids(순위대로 나열된 article_id 리스트) 안에서
    relevant_ids 중 가장 먼저(가장 좋은 순위로) 나오는 위치를 찾는다.
    없으면 None.
    """
    for rank, aid in enumerate(ordered_ids, start=1):
        if aid in relevant_ids:
            return rank
    return None


all_query_results = []

for prepared in prepared_items:

    field = prepared["field"]
    query = prepared["query"]
    relevant_ids = set(prepared["relevant_ids"])
    top10 = prepared["top10"]  # (id, title, content, source_url, distance)

    print("\n" + "=" * 80)
    print(f"[{field}] {query}")
    print("=" * 80)

    # --------------------------------------
    # 벡터 검색 순위 (원래 순서 그대로)
    # --------------------------------------
    vector_ids = [row[0] for row in top10]

    print("\n[Vector 순위]")
    for rank, row in enumerate(top10, start=1):
        aid, title = row[0], row[1]
        mark = "<- 정답" if aid in relevant_ids else ""
        print(f"{rank:2d}위 | id={aid} | {title} {mark}")

    # --------------------------------------
    # 리랭커 재정렬
    # --------------------------------------
    rerank_pairs = [[query, row[2]] for row in top10]
    scores = reranker.compute_score(rerank_pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    reranked = sorted(zip(top10, scores), key=lambda x: x[1], reverse=True)
    reranked_ids = [row[0] for row, _ in reranked]

    print("\n[Reranker 순위]")
    for rank, (row, score) in enumerate(reranked, start=1):
        aid, title = row[0], row[1]
        mark = "<- 정답" if aid in relevant_ids else ""
        print(f"{rank:2d}위 | id={aid} | score={score:.4f} | {title} {mark}")

    # --------------------------------------
    # Rank / MRR
    # --------------------------------------
    vector_rank = find_rank(vector_ids, relevant_ids)
    reranker_rank = find_rank(reranked_ids, relevant_ids)

    vector_mrr = (1 / vector_rank) if vector_rank else 0
    reranker_mrr = (1 / reranker_rank) if reranker_rank else 0

    # --------------------------------------
    # Recall@10 / Precision@10
    # (앞서 설명했듯, 후보 집합 자체는 안 바뀌므로 전/후 동일)
    # --------------------------------------
    hit_count = len(set(vector_ids) & relevant_ids)
    recall_at_10 = 1 if hit_count > 0 else 0
    precision_at_10 = hit_count / 10

    print("\n[결과]")
    print(f"Vector Rank   : {vector_rank if vector_rank else '없음(10위 밖)'}")
    print(f"Reranker Rank : {reranker_rank if reranker_rank else '없음(10위 밖)'}")
    print(f"Vector MRR    : {vector_mrr:.4f}")
    print(f"Reranker MRR  : {reranker_mrr:.4f}")
    print(f"Recall@10     : {recall_at_10} (전/후 동일)")
    print(f"Precision@10  : {precision_at_10:.2f} (전/후 동일)")

    all_query_results.append(
        {
            "field": field,
            "query": query,
            "relevant_ids": list(relevant_ids),
            "vector_rank": vector_rank,
            "reranker_rank": reranker_rank,
            "vector_mrr": vector_mrr,
            "reranker_mrr": reranker_mrr,
            "recall@10": recall_at_10,
            "precision@10": precision_at_10,
        }
    )


# ==========================================
# STEP 8. 최종 평균
# ==========================================

def average(values):
    return sum(values) / len(values) if values else 0


avg_vector_mrr = average([r["vector_mrr"] for r in all_query_results])
avg_reranker_mrr = average([r["reranker_mrr"] for r in all_query_results])
avg_recall_10 = average([r["recall@10"] for r in all_query_results])
avg_precision_10 = average([r["precision@10"] for r in all_query_results])

print("\n\n" + "=" * 80)
print(f"전체 평균 ({len(all_query_results)}개 쿼리)")
print("=" * 80)
print(f"MRR       : {avg_vector_mrr:.4f} -> {avg_reranker_mrr:.4f}")
print(f"Recall@10 : {avg_recall_10:.4f} (전/후 동일)")
print(f"Precision@10 : {avg_precision_10:.4f} (전/후 동일)")
print(f"\nMRR 개선폭: {avg_reranker_mrr - avg_vector_mrr:+.4f}")

summary = {
    "num_queries": len(all_query_results),
    "mrr": {"vector": avg_vector_mrr, "reranker": avg_reranker_mrr},
    "recall@10": avg_recall_10,
    "precision@10": avg_precision_10,
    "details": all_query_results,
}

with open("article_id_gold_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

cur.close()
conn.close()

print("\n결과 파일: article_id_gold_evaluation.json")