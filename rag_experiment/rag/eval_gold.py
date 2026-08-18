"""
골든 데이터셋 기반 검색 성능 평가

gold_candidates.md에서 [x] 체크된 질문-정답 쌍만 읽어서,
실제로 그 질문을 검색했을 때 정답 기사가 몇 위로 나오는지 측정한다.

self-retrieval 평가(eval_all_report.py 파트 B)와의 차이:
    self-retrieval은 "기사 제목 그대로"로 검색했지만,
    이건 사람이 검수한 "자연스러운 질문"으로 검색한다.
    -> 실제 사용자 검색 상황에 훨씬 더 가까운 평가.
"""

import re

from rag.retriever import retrieve_documents


def parse_gold_md(path="gold_candidates.md"):
    """
    gold_candidates.md에서 [x] 체크된 항목만 파싱해서
    {question, expected_title, source_url} 리스트로 반환한다.
    """

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"- \[[xX]\] 질문: (.+?)\n\s+정답: (.+?)\n\s+URL: (\S+)"
    )

    matches = pattern.findall(content)

    gold_set = []

    for question, expected_title, source_url in matches:

        question = question.strip().strip('"')

        gold_set.append({
            "question": question,
            "expected_title": expected_title.strip(),
            "source_url": source_url.strip(),
        })

    return gold_set


def evaluate_gold_set(path="gold_candidates.md"):

    gold_set = parse_gold_md(path)

    total = len(gold_set)

    print("=" * 70)
    print(f"골든 데이터셋 평가 (체크된 질문 {total}개)")
    print("=" * 70)

    rows = []

    for index, item in enumerate(gold_set, start=1):

        question = item["question"]
        url = item["source_url"]

        results = retrieve_documents(query=question, top_k=20)

        values = [r.get("source_url") for r in results]
        rank = values.index(url) + 1 if url in values else None

        score = (1 / rank) if rank else 0.0

        rows.append({
            "question": question,
            "expected_title": item["expected_title"],
            "rank": rank,
            "score": score,
        })

        rank_display = f"{rank}위" if rank else "20위 밖"
        print(f"[{index}/{total}] {question}")
        print(f"    -> {rank_display} (점수: {score:.3f}) | 정답: {item['expected_title']}\n")

    rows.sort(key=lambda r: r["score"])

    hit_at_5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5)
    hit_at_10 = sum(1 for r in rows if r["rank"] and r["rank"] <= 10)
    recall_at_5 = hit_at_5 / total
    recall_at_10 = hit_at_10 / total
    mrr = sum(r["score"] for r in rows) / total

    print("\n" + "=" * 70)
    print("표 (점수 낮은 순)")
    print("=" * 70)
    print()
    print("질문 | 정답 기사 | 순위 | 점수")
    print("---|---|---|---")
    for row in rows:
        rank_display = f"{row['rank']}위" if row["rank"] else "못 찾음"
        print(f"{row['question']} | {row['expected_title']} | {rank_display} | {row['score']:.3f}")

    print("\n" + "=" * 70)
    print("최종 요약 (골든 데이터셋 기반)")
    print("=" * 70)
    print(f"전체 질문 수: {total}개")
    print(f"Recall@5:  {recall_at_5:.3f}")
    print(f"Recall@10: {recall_at_10:.3f}")
    print(f"MRR:       {mrr:.3f}")

    return rows


if __name__ == "__main__":
    evaluate_gold_set()