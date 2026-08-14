"""인사이트+비즈니스 시사점 도출 도구 (구현 항목 #4)."""
from .llm_client import call_llm, parse_json_response

SYSTEM_PROMPT = """당신은 빅테크/AI 업계 애널리스트다. 주어진 주제별 요약들을 종합해
핵심 인사이트 3~5개와 비즈니스 시사점을 도출한다.
각 인사이트는 반드시 근거가 된 원문 URL을 포함해야 한다.
반드시 아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
{
  "insights": [
    {"text": "인사이트 문장", "source_url": "근거 URL"}
  ],
  "business_implication": "비즈니스 시사점 문단"
}
```
"""


def extract_insights(summaries: list[dict]) -> dict:
    """주제별 요약 리스트 -> {insights, business_implication} 형태의 인사이트."""
    if not summaries:
        return {"insights": [], "business_implication": "오늘 수집된 신규 기사가 없습니다."}

    summary_text = "\n\n".join(
        f"주제: {s['topic_title']}\n요약: {s['summary']}\n출처: {', '.join(s.get('source_urls', []))}"
        for s in summaries
    )
    user_prompt = f"다음 요약들을 바탕으로 인사이트와 비즈니스 시사점을 뽑아줘:\n\n{summary_text}"

    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    return parse_json_response(raw)
