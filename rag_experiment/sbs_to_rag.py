"""
SBS 뉴스 → RAG DB 적재

SBS RSS
→ 원문 수집
→ 텍스트 정제
→ Chunk 생성
→ Embedding
→ PostgreSQL + pgvector 저장
"""

import psycopg2

from sbs_fetch import fetch_sbs_rss, fetch_sbs_article
from news_preprocess import clean_text, chunk_text
from rag.indexer import index_chunks


DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_test",
    "user": "rag_user",
    "password": "rag_password",
}


def article_exists(source_url):
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

        # Embedding + DB 저장
        try:
            saved = index_chunks(
                title=title,
                source_url=url,
                chunks=chunks
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