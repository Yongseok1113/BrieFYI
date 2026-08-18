"""요약/검증/판정용 LLM 호출. data_pipeline/src/data_pipeline/llm_client.py와 동일한
패턴(공용 call_llm, provider별 lazy 클라이언트)을 최소 기능만 가져와 새로 만든다 —
data_pipeline/을 import하지 않는다(별도 배포 이미지). groq/anthropic 두 provider만
지원한다(hf는 이 용도에 불필요).
"""
from __future__ import annotations

import json
import re

from config import config

_groq_client = None
_anthropic_client = None


class LLMError(RuntimeError):
    pass


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        if not config.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _call_groq(system: str, user: str, max_tokens: int) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_llm(system: str, user: str, *, max_tokens: int = 1500, provider: str | None = None) -> str:
    # 모듈 레벨이 아니라 호출 시점에 매핑을 구성한다 — 그래야 테스트에서
    # patch("rag_latest.llm_client._call_groq", ...)로 교체한 mock이 실제로 쓰인다
    # (모듈 레벨 dict는 import 시점에 원본 함수 참조를 그대로 캡처해버려 patch가 무력화된다).
    _PROVIDERS = {"groq": _call_groq, "anthropic": _call_anthropic}
    provider = provider or config.RAG_SUMMARIZE_PROVIDER
    call_fn = _PROVIDERS.get(provider)
    if call_fn is None:
        raise LLMError(f"지원하지 않는 provider: {provider} (groq/anthropic 중 하나)")
    try:
        return call_fn(system, user, max_tokens)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider별 예외 타입이 달라 폭넓게 잡는다
        raise LLMError(f"LLM 호출 실패 ({provider}): {exc}") from exc


def parse_json_response(text: str) -> dict | list:
    """```json 코드블록 우선, 없으면 첫 '{'/'['부터 raw_decode — data_pipeline/llm_client.py와
    동일한 관용구(모델이 JSON 앞뒤에 설명을 덧붙이는 경우까지 안전하게 처리)."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    payload = (match.group(1) if match else text).strip()

    starts = [i for i in (payload.find("{"), payload.find("[")) if i != -1]
    start = min(starts) if starts else 0

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(payload[start:])
    return obj
