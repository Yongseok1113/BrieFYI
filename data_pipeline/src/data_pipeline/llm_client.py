"""LLM 클라이언트. 파인튜닝 안 한 기본 모델을 무료 API로 호출하는 것이 기본값이다
(LLM_PROVIDER=groq) — Claude 비용/약관 이슈 없이 프롬프트 엔지니어링만으로 학습
데이터를 만들기 위함. data_pipeline은 별도 컨테이너라 tools/ 쪽 모듈을 import하지
않고 자체 구현을 둔다.

Groq가 기본값이다: HF Inference Providers는 무료 계정 월 $0.10 크레딧뿐이고 그마저
카드 등록이 있어야 라우팅이 열려서 실제 운영엔 못 쓴다(자세한 경위는 config.py 주석
참고). hf/anthropic 경로는 남겨뒀으니 필요하면 DATA_PIPELINE_LLM_PROVIDER로 전환.

모든 호출은 rate_limiter.py를 거치고, 429/5xx는 지수 백오프로 재시도한다.
"""
from __future__ import annotations

import json
import re
import time

from .config import config
from .rate_limiter import RateLimiter

_limiter = RateLimiter(config.RATE_LIMIT_MAX_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
_groq_client = None
_hf_client = None
_anthropic_client = None


class LLMError(RuntimeError):
    pass


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY가 설정되지 않았습니다 (.env 확인, https://console.groq.com/keys 에서 발급)")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient

        if not config.HF_API_TOKEN:
            raise LLMError("HF_API_TOKEN이 설정되지 않았습니다 (.env 확인)")
        # provider를 명시적으로 고정한다 — "auto"로 두면 계정에서 활성화 안 한
        # provider로 라우팅을 시도하다 model_not_supported로 실패하는 경우가 있었다.
        _hf_client = InferenceClient(
            model=config.HF_MODEL_ID, token=config.HF_API_TOKEN, provider=config.DATA_PIPELINE_HF_PROVIDER
        )
    return _hf_client


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


def _call_hf(system: str, user: str, max_tokens: int) -> str:
    client = _get_hf_client()
    response = client.chat_completion(
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
    """레이트 리미터 + 재시도를 포함한 공통 LLM 호출 지점."""
    provider = provider or config.LLM_PROVIDER
    providers = {"groq": _call_groq, "hf": _call_hf, "anthropic": _call_anthropic}
    call_fn = providers.get(provider)
    if call_fn is None:
        raise LLMError(f"지원하지 않는 provider: {provider} (groq/hf/anthropic 중 하나)")

    last_exc: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        _limiter.acquire()
        try:
            return call_fn(system, user, max_tokens)
        except Exception as exc:  # noqa: BLE001 - 429/5xx를 포함해 provider별 예외 타입이 달라 폭넓게 잡는다
            last_exc = exc
            message = str(exc).lower()
            retryable = any(code in message for code in ("429", "rate limit", "503", "502", "timeout"))
            if not retryable or attempt == config.LLM_MAX_RETRIES - 1:
                raise LLMError(f"LLM 호출 실패 ({provider}): {exc}") from exc
            backoff = min(60, 2**attempt)
            time.sleep(backoff)

    raise LLMError(f"LLM 호출 실패 ({provider}): {last_exc}")


def parse_json_response(text: str) -> dict | list:
    """```json 코드블록으로 감싸 답하는 경우, 그리고 JSON 앞/뒤에 모델이 부연설명을
    덧붙이는 경우(Groq llama-3.3에서 자주 관찰됨)까지 포함해 JSON을 안전하게 추출한다.
    json.loads는 문자열 전체가 하나의 JSON이어야 해서 뒤에 아무 텍스트만 붙어도
    "Extra data" 에러가 나므로, raw_decode로 첫 JSON 값만 파싱하고 나머지는 버린다.
    """
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    payload = (match.group(1) if match else text).strip()

    # 앞에 설명 문구가 붙어 JSON이 중간부터 시작하는 경우 첫 '{'/'['부터 파싱한다.
    starts = [i for i in (payload.find("{"), payload.find("[")) if i != -1]
    start = min(starts) if starts else 0

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(payload[start:])
    return obj
