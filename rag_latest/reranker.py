"""검색 결과 재순위화 (cross-encoder reranker).

rag_experiment/rag/reranker.py의 BAAI/bge-reranker-v2-m3 접근을 rag_latest.retriever의
retrieve() 출력 행 형태(title/text 키)에 맞춰 재작성했다. retrieve()가 반환한 후보를
받아 재정렬만 하고, 검색 자체(vector/text/hybrid)는 건드리지 않는다.

로컬 모델을 GPU/CPU에 올리므로 embed.py(HF Inference API, 로컬 메모리 사용 안 함)와
달리 별도 프로세스 메모리를 쓴다. reranker와 embedding 모델을 동시에 올리지 않도록
호출 측(agent_tool.py)이 unload_reranker()로 정리한다.
"""
from __future__ import annotations

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    return _reranker


def unload_reranker() -> None:
    global _reranker
    if _reranker is not None:
        del _reranker
        _reranker = None
        import gc

        gc.collect()


def rerank(query: str, rows: list[dict], top_k: int | None = None) -> list[dict]:
    """retrieve()가 반환한 후보 목록을 cross-encoder 점수로 재정렬한다.

    row["score"](RRF/boost 점수)는 건드리지 않고 row["rerank_score"]를 새로 추가한 뒤
    그 값으로 재정렬한다 — 재정렬 전 순위(row["pre_rerank_rank"])도 함께 남겨 비교할 수
    있게 한다.
    """
    if not rows:
        return []

    reranker = get_reranker()
    pairs = [(query, f"{row['title']}\n{row['text']}") for row in rows]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    reranked = [
        {**row, "pre_rerank_rank": i + 1, "rerank_score": float(score)}
        for i, (row, score) in enumerate(zip(rows, scores))
    ]
    reranked.sort(key=lambda row: (-row["rerank_score"], row["chunk_id"]))
    return reranked[:top_k] if top_k is not None else reranked
