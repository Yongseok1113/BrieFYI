"""
SBS 뉴스 → RAG DB 적재

SBS RSS
→ 원문 수집
→ 텍스트 정제
→ Chunk 생성
→ LLM 메타데이터 분류
→ Embedding
→ PostgreSQL + pgvector 저장

[변경사항]
Chunk를 만든 다음 index_chunks()를 부르기 "직전"에
classify.py의 classify_article()을 호출해서
category/domain/entities/event를 미리 뽑아내고,
그 결과를 index_chunks()에 같이 넘기도록 변경함.

주의: classify_article()은 "기사 1개당 1번만" 호출한다.
"""

import psycopg2

from sbs_fetch import fetch_sbs_rss, fetch_sbs_article
from news_preprocess import clean_text, chunk_text
from rag.indexer import index_chunks
from rag.classify import classify_article   # ← 새로 추가된 import


DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_test",
    "user": "rag_user",
    "password": "rag_password",
}


def article_exists(source_url):
    """
    이미 저장된 기사인지 URL로 중복 확인한다.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM rag_documents
                WHERE source_url = %s
            )
            """,
            (source_url,)
        )

        return cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()


def main():

    print("=" * 60)
    print("SBS → RAG 데이터 적재")
    print("=" * 60)

    articles = fetch_sbs_rss(max_results=5)

    print(f"\nSBS RSS 기사 {len(articles)}개 수집")

    total_articles = 0
    total_chunks = 0
    skipped_articles = 0

    for index, article in enumerate(articles, start=1):

        title = article["title"]
        url = article["url"]

        print("\n" + "-" * 60)
        print(f"[{index}/{len(articles)}] {title}")

        # 중복 확인
        if article_exists(url):
            print("이미 저장된 기사 → 건너뜀")
            skipped_articles += 1
            continue

        # SBS 원문 수집
        try:
            content = fetch_sbs_article(url)
        except Exception as e:
            print(f"원문 수집 실패: {e}")
            continue

        if not content:
            print("원문이 비어 있음 → 건너뜀")
            continue

        print(f"원문 길이: {len(content)}자")

        # 텍스트 정제 
        cleaned_content = clean_text(content)

        # Chunk 생성
        chunks = chunk_text(
            cleaned_content,
            chunk_size=500,
            chunk_overlap=50
        )

        print(f"생성된 Chunk: {len(chunks)}개")

        if not chunks:
            print("Chunk 없음 → 건너뜀")
            continue

        # 기사 1개당 1번만 분류 요청을 보낸다.
        # (chunk 개수만큼 여러 번 부르지 않도록,
        #  chunk로 쪼개기 전의 "정제된 원문 전체"를 기준으로 분류)
        print("메타데이터 분류 중 (LLM 호출)...")

        metadata = classify_article(
            title=title,
            content=cleaned_content
        )

        print(
            f"분류 결과: "
            f"category={metadata['category']}, "
            f"domain={metadata['domain']}, "
            f"entities={metadata['entities']}, "
            f"event={metadata['event']}"
        )

        # Embedding + DB 저장
        try:
            saved = index_chunks(
                title=title,
                source_url=url,
                chunks=chunks,
                # 방금 분류한 metadata를 그대로 index_chunks에 전달
                category=metadata["category"],
                domain=metadata["domain"],
                entities=metadata["entities"],
                event=metadata["event"],
            )

            print(f"DB 저장 완료: {saved}개 Chunk")

            total_articles += 1
            total_chunks += saved

        except Exception as e:
            print(f"DB 저장 실패: {e}")

    print("\n" + "=" * 60)
    print("SBS → RAG 적재 완료")
    print("=" * 60)

    print(f"새로 저장한 기사: {total_articles}개")
    print(f"새로 저장한 Chunk: {total_chunks}개")
    print(f"중복으로 건너뛴 기사: {skipped_articles}개")


if __name__ == "__main__":
    main()