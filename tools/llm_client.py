"""Anthropic 클라이언트 공용 헬퍼. summarize/insight 도구가 공유해서 사용한다."""
import json
import re

from anthropic import Anthropic

from config import config

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def call_llm(system: str, user: str, max_tokens: int = 2000) -> str:
    client = get_client()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def parse_json_response(text: str) -> dict | list:
    """모델이 ```json 코드블록으로 감싸 답하는 경우를 포함해 JSON을 안전하게 추출한다."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    payload = match.group(1) if match else text
    return json.loads(payload)
