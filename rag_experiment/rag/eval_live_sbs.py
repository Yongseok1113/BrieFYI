"""
RAG 검색 성능 평가 - 실제 SBS 신규 기사로 Self-Retrieval 평가 (MRR)

"한 번도 DB에 없던 진짜 SBS 기사"를 새로 가져와서
분류→저장까지 실제 파이프라인 그대로 돌린 다음,
그 기사의 "제목"을 검색어로 써서
그 기사 본인이 검색 결과 몇 위로 나오는지 확인하고
MRR이라는 소수점 점수로 정리한다.

새로운 기사가 들어와도, 기사 제목 자체를 질문 삼아
"내가 진짜 나를 찾아내는가?"를 확인하는 방식으로 평가한다.

MRR (Mean Reciprocal Rank)이 뭔가?
질문마다 "1 / 정답 순위"를 계산해서 평균낸 것.
    1위로 찾음  → 1 / 1 = 1.0
    2위로 찾음  → 1 / 2 = 0.5
    3위로 찾음  → 1 / 3 = 0.33
    못 찾음     → 0
이렇게 소수점 점수로 나와서, "1위/2위" 같은 순위 표기보다
"검색 품질이 전체적으로 몇 점이다"를 한 숫자로 보여주기 좋다.
"""

from sbs_fetch import fetch_sbs_rss, fetch_sbs_article
from news_preprocess import clean_text, chunk_text
from sbs_to_rag import article_exists

from rag.classify import classify_article
from rag.indexer import index_chunks
from rag.retriever import retrieve_documents


def collect_new_sbs_articles(max_results: int = 10) -> list:
    """
    SBS RSS에서 기사를 가져오되,
    이미 DB에 있는(=예전에 저장한 적 있는) 기사는 걸러내고
    "진짜 처음 보는" 기사만 골라서 실제로 분류+저장까지 진행한다.

    Returns:
        새로 저장한 기사 정보 리스트.
        각 항목: {"title": ..., "source_url": ...}
    """

    print("SBS RSS에서 기사 가져오는 중...")

    articles = fetch_sbs_rss(max_results=max_results)

    newly_indexed = []

    for article in articles:

        title = article["title"]
        url = article["url"]

        # --------------------------------------
        # 이미 DB에 있는 기사면 건너뛴다.
        # (sbs_to_rag.py에서 쓰던 것과 동일한 중복 확인 함수 재사용)
        # --------------------------------------
        if article_exists(url):
            continue

        try:
            content = fetch_sbs_article(url)
        except Exception as e:
            print(f"  원문 수집 실패 ({title}): {e}")
            continue

        if not content:
            continue

        cleaned_content = clean_text(content)

        chunks = chunk_text(cleaned_content, chunk_size=500, chunk_overlap=50)

        if not chunks:
            continue

        # 기사 1개당 1번만 분류 (기존 파이프라인과 동일한 원칙)
        metadata = classify_article(title=title, content=cleaned_content)

        index_chunks(
            title=title,
            source_url=url,
            chunks=chunks,
            category=metadata["category"],
            domain=metadata["domain"],
            entities=metadata["entities"],
            event=metadata["event"],
        )

        print(f"  새로 저장: {title}")

        newly_indexed.append({"title": title, "source_url": url})

    return newly_indexed


def find_rank_by_source_url(results: list, target_url: str):
    """
    검색 결과 리스트 안에서, source_url이 target_url과
    일치하는 첫 번째(=가장 순위 높은) 위치를 찾는다.

    기사 하나가 여러 chunk로 쪼개져 저장돼 있으므로,
    "제목 일치"가 아니라 "source_url 일치"로 찾는다.
    (같은 기사의 chunk 여러 개가 상위권에 다 몰려 있어도,
     그중 제일 순위 높은 것 하나만 카운트하면 된다)
    """

    for index, result in enumerate(results, start=1):

        if result.get("source_url") == target_url:
            return index

    return None


def evaluate_self_retrieval_mrr(max_results: int = 10):
    """
    새로 수집한 SBS 기사들에 대해 self-retrieval 평가를 수행하고
    MRR을 계산한다.
    """

    print("=" * 70)
    print("Self-Retrieval 평가 (실제 신규 SBS 기사)")
    print("=" * 70)

    new_articles = collect_new_sbs_articles(max_results=max_results)

    if not new_articles:
        print("\n새로 수집된 기사가 없습니다 (전부 이미 DB에 있음).")
        print("sbs_to_rag.py를 이미 여러 번 돌려서 최근 RSS 기사가")
        print("전부 중복 처리된 것으로 보입니다. max_results를 늘리거나")
        print("나중에 RSS가 갱신된 뒤 다시 시도해주세요.")
        return

    print(f"\n새로 저장된 기사: {len(new_articles)}개")
    print("\n각 기사 제목을 검색어로 써서 self-retrieval 테스트 중...")

    reciprocal_ranks = []

    detail_rows = []

    for article in new_articles:

        title = article["title"]
        url = article["source_url"]

        # 제목 그대로를 검색어로 사용.
        # top_k는 넉넉하게 20으로 잡아서, 20위 밖으로 밀리면
        # "사실상 못 찾은 것"으로 처리한다.
        results = retrieve_documents(query=title, top_k=20)

        rank = find_rank_by_source_url(results, url)

        # --------------------------------------
        # Reciprocal Rank 계산.
        # rank가 None(못 찾음)이면 0점 처리.
        # --------------------------------------
        reciprocal_rank = (1 / rank) if rank else 0

        reciprocal_ranks.append(reciprocal_rank)

        rank_display = f"{rank}위" if rank else "20위 밖"

        print(f"\n  제목: {title}")
        print(f"  검색 결과 순위: {rank_display} (점수: {reciprocal_rank:.3f})")

        detail_rows.append(
            {
                "title": title,
                "rank": rank,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    # --------------------------------------
    # MRR = 전체 reciprocal rank의 평균
    # --------------------------------------
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 70)
    print("요약 표")
    print("=" * 70)
    print()
    print("기사 제목 | 검색 순위 | 점수")
    print("---|---|---")

    for row in detail_rows:
        rank_display = f"{row['rank']}위" if row["rank"] else "20위 밖"
        print(f"{row['title']} | {rank_display} | {row['reciprocal_rank']:.3f}")

    print()
    print(f"MRR (Mean Reciprocal Rank): {mrr:.3f}")

    return {"mrr": mrr, "details": detail_rows}


if __name__ == "__main__":
    evaluate_self_retrieval_mrr(max_results=10)

    print("\n" + "=" * 70)
    print("평가 완료")
    print("=" * 70)