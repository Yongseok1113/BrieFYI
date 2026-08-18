import json
import psycopg2


# -----------------------------------
# DB 연결
# -----------------------------------
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password",
)

cur = conn.cursor()


# -----------------------------------
# 평가할 키워드
# -----------------------------------
keywords = [
    "삼성전자",
    "반도체",
    "축구",
    "이정후",
    "토미 존",
    "호날두",
    "아스날",
]


gold = []


# -----------------------------------
# 키워드별 관련 기사 검색
# -----------------------------------
for keyword in keywords:

    pattern = f"%{keyword}%"

    query = """
        SELECT
            id,
            title
        FROM rag_documents
        WHERE title ILIKE %s
        ORDER BY published_at DESC NULLS LAST, id DESC
        LIMIT 10
    """

    cur.execute(query, (pattern,))

    rows = cur.fetchall()

    # 중복 제목 제거
    titles = []
    seen = set()

    for article_id, title in rows:

        if not title:
            continue

        if title not in seen:
            titles.append(title)
            seen.add(title)

    gold.append(
        {
            "keyword": keyword,
            "relevant_titles": titles,
        }
    )


# -----------------------------------
# JSON 저장
# -----------------------------------
with open(
    "gold_keyword_candidates.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        gold,
        f,
        ensure_ascii=False,
        indent=2,
    )


# -----------------------------------
# 종료
# -----------------------------------
cur.close()
conn.close()


print("gold_keyword_candidates.json 생성 완료")
print(f"키워드 수: {len(gold)}")

for item in gold:
    print(
        f"- {item['keyword']}: "
        f"{len(item['relevant_titles'])}개 정답 후보 기사"
    )