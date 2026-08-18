"""검색 품질 평가 (Recall@k / MRR).

rag_experiment의 eval_*.py 계열(7개 스크립트)을 self-retrieval 방법론 하나로 통합해
rag_latest.retriever.retrieve()를 대상으로 재작성했다. 합성 데이터셋·LLM 골드셋 생성은
실제 색인 데이터가 충분히 쌓인 뒤에야 의미가 있어 이번 이식 범위에서 제외했다
(rag_experiment/gold_eval_result.md, recall_result.md 참고).

DB + HF embedding API가 필요한 수동 리포트 도구다. CI에서 자동 실행하지 않는다.

    from rag_latest.eval import evaluate_self_retrieval
    evaluate_self_retrieval(article_ids=[19, 20, 21], top_k_values=(5, 10))
"""
from __future__ import annotations

from . import db, retriever


def reciprocal_rank(ranked_ids: list[int], target_id: int) -> float:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id == target_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[int], target_id: int, k: int) -> bool:
    return target_id in ranked_ids[:k]


def evaluate_self_retrieval(
    article_ids: list[int] | None = None,
    top_k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """각 기사의 제목으로 검색했을 때 그 기사 자신의 chunk가 몇 위에 나오는지 측정한다.

    recall_result.md의 Part B(231개 실제 기사, Recall@5=0.961/Recall@10=0.991/MRR=0.844)와
    같은 방법론이다 — "뚜렷이 구별되는 기사는 거의 항상 상위에 나오지만, 같은 사건을 다룬
    근접 중복 기사가 많을수록 순위가 밀린다"는 패턴이 여기서도 드러나는지 확인하는 용도다.
    """
    if article_ids is None:
        # "인덱싱된 기사 전부"를 자동으로 고르는 쿼리는 db.py에 없다(색인 안 된 기사를
        # 찾는 load_unindexed_article_ids()의 반대 개념). 새 쿼리를 추가하는 대신, 평가
        # 대상은 호출자가 이미 index_articles()로 색인한 ID 목록을 명시적으로 넘기게 한다.
        raise ValueError(
            "article_ids를 명시적으로 넘겨야 한다 (예: 이미 index_articles()로 색인된 ID 목록)."
        )

    articles = {row["id"]: row for row in db.load_articles(article_ids)}
    max_k = max(top_k_values)

    reciprocal_ranks = []
    hits_by_k = {k: 0 for k in top_k_values}
    evaluated = 0

    for article_id in article_ids:
        article = articles.get(article_id)
        if article is None or not article.get("title"):
            continue
        rows = retriever.retrieve(article["title"], top_k=max_k, search_mode="hybrid")
        ranked_article_ids = [row["article_id"] for row in rows]

        evaluated += 1
        reciprocal_ranks.append(reciprocal_rank(ranked_article_ids, article_id))
        for k in top_k_values:
            if recall_at_k(ranked_article_ids, article_id, k):
                hits_by_k[k] += 1

    if evaluated == 0:
        return {"evaluated": 0, "mrr": 0.0, "recall_at_k": {k: 0.0 for k in top_k_values}}

    return {
        "evaluated": evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
        "recall_at_k": {k: hits_by_k[k] / evaluated for k in top_k_values},
    }
