"""기사 제목과 설명에서 4-Layer의 Entity/Event metadata를 추출한다."""
import json

from .llm_client import call_llm, parse_json_response


SYSTEM_PROMPT = """당신은 뉴스 기사 metadata 분류기다.
기사에서 핵심적으로 다뤄지는 Entity와 Event만 추출한다.

- entities: 회사, 기관, 인물, 제품 등의 고유명사. 최대 3개.
- events: 핵심 사건이나 행동을 나타내는 짧은 명사구. 최대 2개.
- 기사에 근거하지 않은 항목을 만들지 않는다.
- 적절한 항목이 없으면 빈 배열을 사용한다.
- 반드시 아래 JSON 객체만 반환한다. 다른 설명은 붙이지 않는다.

```json
{
  "entities": ["NVIDIA", "OpenAI"],
  "events": ["투자"]
}
```
"""


def _normalize_items(payload: dict, key: str, max_items: int) -> list[str]:
    items = payload.get(key)
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ValueError(f"{key}는 문자열 배열이어야 합니다.")

    normalized = list(dict.fromkeys(item.strip() for item in items if item.strip()))
    if len(normalized) > max_items:
        raise ValueError(f"{key}는 최대 {max_items}개여야 합니다: {len(normalized)}개")
    return normalized


def extract_article_topics(title: str, description: str | None) -> dict[str, list[str]]:
    """기사 한 건에서 Entity 최대 3개와 Event 최대 2개를 추출한다."""
    title = title.strip()
    description = (description or "").strip()
    if not title:
        raise ValueError("title은 비어 있을 수 없습니다.")

    article = json.dumps(
        {"title": title, "description": description},
        ensure_ascii=False,
        indent=2,
    )
    raw = call_llm(
        SYSTEM_PROMPT,
        f"다음 기사를 분류해줘:\n\n{article}",
        max_tokens=500,
    )
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM 응답은 JSON 객체여야 합니다.")

    return {
        "entities": _normalize_items(parsed, "entities", max_items=3),
        "events": _normalize_items(parsed, "events", max_items=2),
    }
