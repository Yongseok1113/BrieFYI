"""
대량 SBS 기사 수집 (Recall 대규모 테스트용)
"""

from sbs_fetch import fetch_sbs_rss_multi, fetch_sbs_article
from news_preprocess import clean_text, chunk_text
from sbs_to_rag import article_exists

from rag.indexer import index_chunks


def bulk_collect(max_results_per_section=30):

    print("=" * 60)
    print("SBS 대량 수집 시작 (분류 없이, 비용 0원)")
    print("=" * 60)

    print("\n섹션별 RSS 확인 중...")

    articles = fetch_sbs_rss_multi(
        max_results_per_section=max_results_per_section
    )

    print(f"\n전체 수집 대상: {len(articles)}개")

    saved_articles = 0
    saved_chunks = 0
    skipped = 0
    failed = 0

    for index, article in enumerate(articles, start=1):

        title = article["title"]
        url = article["url"]
        section = article["section"]

        print(f"\n[{index}/{len(articles)}] ({section}) {title}")

        if article_exists(url):
            print("  이미 저장됨 -> 건너뜀")
            skipped += 1
            continue

        try:
            content = fetch_sbs_article(url)
        except Exception as e:
            print(f"  원문 수집 실패: {e}")
            failed += 1
            continue

        if not content:
            print("  원문 없음 -> 건너뜀")
            failed += 1
            continue

        cleaned_content = clean_text(content)

        chunks = chunk_text(cleaned_content, chunk_size=500, chunk_overlap=50)

        if not chunks:
            print("  Chunk 없음 -> 건너뜀")
            failed += 1
            continue

        # --------------------------------------
        # indexer.py가 이미 이 값들을 선택사항으로 처리하도록
        #  돼있어서 에러 없이 정상 저장된다
        # --------------------------------------

        try:
            saved = index_chunks(
                title=title,
                source_url=url,
                chunks=chunks,
                # category, domain, entities, event는 전달 안 함 -> 전부 None
            )

            saved_articles += 1
            saved_chunks += saved

        except Exception as e:
            print(f"  DB 저장 실패: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("대량 수집 완료")
    print("=" * 60)
    print(f"새로 저장한 기사: {saved_articles}개")
    print(f"새로 저장한 Chunk: {saved_chunks}개")
    print(f"이미 있어서 건너뜀: {skipped}개")
    print(f"실패: {failed}개")


if __name__ == "__main__":
    bulk_collect(max_results_per_section=30)