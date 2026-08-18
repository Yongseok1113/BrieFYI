"""
골든 데이터셋 생성 (LLM 기반, 자연스러운 질문)

DB에 있는 기사를 무작위로 30개를 뽑아,
각 기사에 대해 "실제 사람이 검색할 법한 자연스러운 질문"을
Claude API로 만든다.

이 초안을 gold_candidates.md 파일로 저장하고,
사람이 열어서 "말이 되는 것만" 골라 [x] 표시하면
골든 데이터셋(질문-정답 쌍)이 된다.
"""

import json
import os
import random

import psycopg2
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv

from rag.db_config import DB_CONFIG

load_dotenv(find_dotenv())

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

PROMPT = """다음 뉴스 기사를 찾기 위해, 실제 사람이 검색창이나
채팅에 입력할 법한 자연스러운 질문을 1개 만들어줘.

제목: {title}
본문 앞부분: {content}

자연스러운 질문의 예시:
- "삼성전자 이번 실적 어땠어?"
- "요즘 김민석 관련해서 뭔 일 있었어?"
- "거제 산사태 관련 소식 알려줘"
- "이번에 로카르노 영화제에서 상 받은 사람 있어?"

조건:
- 제목을 그대로 잘라 붙이지 말고, 핵심 내용을 자기 말로 풀어써
- 세부 숫자나 고유명사는 남겨도 되지만, 문장 전체는 자연스러운 구어체로
- 한 문장, 20단어 이내

질문만 출력해. 다른 설명 붙이지 마.
"""


def get_random_articles(n=30, min_content_length=100):
    """
    본문이 min_content_length자 이상인 기사 중
    n개를 무작위로 뽑는다.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT title, source_url, MIN(content) as content
            FROM rag_documents
            WHERE source_url LIKE 'https://news.sbs.co.kr%%'
            GROUP BY title, source_url
            HAVING SUM(LENGTH(content)) >= %s
            """,
            (min_content_length,)
        )

        rows = cur.fetchall()

    finally:
        cur.close()
        conn.close()

    sample = random.sample(rows, min(n, len(rows)))

    return [
        {"title": r[0], "source_url": r[1], "content": r[2]}
        for r in sample
    ]


def generate_question(title: str, content: str) -> str:

    prompt = PROMPT.format(title=title, content=(content or "")[:800])

    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def generate_gold_candidates(n=30):

    print(f"기사 {n}개 무작위 샘플링 중...")

    articles = get_random_articles(n)

    print(f"{len(articles)}개 기사에 대해 자연스러운 질문 생성 중...\n")

    candidates = []

    for index, article in enumerate(articles, start=1):

        question = generate_question(article["title"], article["content"])

        candidates.append(
            {
                "question": question,
                "expected_title": article["title"],
                "source_url": article["source_url"],
            }
        )

        print(f"[{index}/{len(articles)}] {question}")
        print(f"    -> 정답 기사: {article['title']}\n")

    # --------------------------------------
    # 마크다운 체크리스트로 저장
    # --------------------------------------

    with open("gold_candidates.md", "w", encoding="utf-8") as f:

        f.write("# 골든 데이터셋 후보 (검수 필요)\n\n")
        f.write("말이 되는 것만 앞에 [x] 표시하고, 이상한 건 그대로 [ ] 두세요.\n\n")

        for c in candidates:
            f.write(f"- [ ] 질문: {c['question']}\n")
            f.write(f"      정답: {c['expected_title']}\n")
            f.write(f"      URL: {c['source_url']}\n\n")

    # 나중에 코드에서 바로 불러 쓸 수 있게 JSON으로도 저장
    with open("gold_candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("완료: gold_candidates.md (검수용) / gold_candidates.json (원본) 생성됨")
    print("=" * 60)


if __name__ == "__main__":
    generate_gold_candidates(n=30)