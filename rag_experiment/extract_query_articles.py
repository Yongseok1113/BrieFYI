import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password",
)

cur = conn.cursor()

targets = {
    "야구": ["야구", "프로야구", "MLB", "KBO"],
    "삼성": ["삼성", "삼성전자", "삼성 라이온즈"],
    "축구": ["축구", "축구협회", "대표팀", "EPL", "K리그"],
    "반도체": ["반도체", "메모리", "D램", "HBM", "AI 반도체"],
}

for category, keywords in targets.items():
    print("\n" + "=" * 80)
    print(f"### {category}")
    print("=" * 80)

    conditions = []
    params = []

    for keyword in keywords:
        conditions.append("""
            (
                title ILIKE %s
                OR content ILIKE %s
                OR keyword ILIKE %s
                OR category ILIKE %s
                OR domain ILIKE %s
            )
        """)

        pattern = f"%{keyword}%"
        params.extend([pattern, pattern, pattern, pattern, pattern])

    query = f"""
        SELECT
            id,
            title,
            keyword,
            category,
            domain,
            event,
            content
        FROM rag_documents
        WHERE {" OR ".join(conditions)}
        ORDER BY published_at DESC
        LIMIT 15
    """

    cur.execute(query, params)
    rows = cur.fetchall()

    print(f"관련 기사 {len(rows)}건\n")

    for i, row in enumerate(rows, 1):
        article_id, title, keyword, article_category, domain, event, content = row

        print(f"[{i}] article_id: {article_id}")
        print(f"제목: {title}")
        print(f"keyword: {keyword}")
        print(f"category: {article_category}")
        print(f"domain: {domain}")
        print(f"event: {event}")

        if content:
            # 원문의 앞부분만 확인
            preview = content.replace("\n", " ")[:700]
            print(f"원문: {preview}")

        print("-" * 80)

cur.close()
conn.close()