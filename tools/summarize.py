"""요약 도구 (구현 항목 #3). 수집된 기사들을 주제별로 묶어 요약한다."""
from .llm_client import call_llm, parse_json_response

SYSTEM_PROMPT = """당신은 뉴스 요약 전문가다. 주어진 기사 목록을 관련 주제끼리 묶고,
각 주제를 200~300자 내외 한국어 요약문으로 정리한다.
반드시 아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
[
  {
    "topic_title": "주제 제목",
    "summary": "요약문",
    "source_urls": ["기사 URL", "..."]
  }
]
```
"""


def summarize_articles(articles: list[dict]) -> list[dict]:
    """원시 기사 리스트 -> 주제별 요약 리스트. 기사가 없으면 빈 리스트를 반환한다."""
    if not articles:
        return []

    article_text = "\n\n".join(
        f"제목: {a['title']}\n설명: {a.get('description', '')}\nURL: {a['url']}"
        for a in articles
    )
    user_prompt = f"다음 기사들을 요약해줘:\n\n{article_text}"

    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    return parse_json_response(raw)
