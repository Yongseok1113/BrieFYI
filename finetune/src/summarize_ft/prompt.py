"""chat template 구성 + 긴 기사 토큰 절단.

tools/summarize.py, tools/insight.py가 쓰는 프롬프트를 학습용 chat 메시지
형태로 그대로 재사용한다. tokenizer는 HF PreTrainedTokenizer를 기대하지만
duck-typing이라 `.encode()`/`.decode()`/`.apply_chat_template()`만 있으면
어떤 것이든 동작하고, tokenizer가 없을 때는 문자 수 기반 근사치로 대체해
GPU/모델 없이도 이 모듈을 단위 테스트할 수 있다.
"""
from __future__ import annotations

from typing import Any, Protocol

SUMMARIZE_SYSTEM_PROMPT = (
    "당신은 뉴스 기사를 한국어로 요약하는 전문 에디터입니다. "
    "핵심만 간결하게, 사실만 담아 요약하세요. 출력은 반드시 JSON이어야 합니다."
)

INSIGHT_SYSTEM_PROMPT = (
    "당신은 여러 뉴스 요약을 바탕으로 비즈니스 인사이트를 도출하는 애널리스트입니다. "
    "근거가 충분하면 인사이트를 2~5개 도출하고(억지로 다 채우지 말고 근거가 뒷받침하는 만큼만), "
    "각 인사이트에는 근거가 된 기사의 source_url을 "
    "반드시 포함하세요. 다만 입력이 특정 산업·시장·전략과 연결되는 구조적 함의가 없는 단발성 "
    "사실(예: 개별 사건·사고, 확정 안 된 루머)이라면 억지로 인사이트를 만들어내지 말고 "
    "insights를 빈 배열로 두고 no_strong_insight를 true로 표시하세요. "
    "출력은 반드시 JSON이어야 합니다."
)


class TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...
    def decode(self, ids: list[int], **kwargs: Any) -> str: ...


def _char_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def truncate_text(text: str, max_tokens: int, tokenizer: TokenizerLike | None = None) -> str:
    """긴 기사 본문을 max_tokens 이내로 자른다.

    tokenizer가 주어지면 실제 토큰 수 기준, 없으면 "토큰당 평균 2.5자"로 근사한다
    (한국어는 토큰당 문자 수가 영어보다 짧은 경향이 있어 보수적으로 잡음).
    """
    if tokenizer is None:
        return _char_truncate(text, max_tokens * 2)
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_summarize_messages(article_title: str, article_text: str, *, max_input_tokens: int = 900,
                              tokenizer: TokenizerLike | None = None) -> list[dict[str, str]]:
    body = truncate_text(article_text, max_input_tokens, tokenizer)
    user = f"# 제목\n{article_title}\n\n# 본문\n{body}\n\n위 기사를 요약해줘."
    return [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_insight_messages(summaries: list[dict[str, Any]], *, max_input_tokens: int = 900,
                            tokenizer: TokenizerLike | None = None) -> list[dict[str, str]]:
    import json

    payload = json.dumps(summaries, ensure_ascii=False)
    payload = truncate_text(payload, max_input_tokens, tokenizer)
    user = f"# 요약 묶음\n{payload}\n\n위 요약들을 바탕으로 비즈니스 인사이트를 3~5개 추출해줘."
    return [
        {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_messages(example: dict[str, Any], *, max_input_tokens: int = 900,
                    tokenizer: TokenizerLike | None = None) -> list[dict[str, str]]:
    """schema.py의 Example dict 하나를 chat message 리스트로 변환하는 공통 진입점."""
    task = example["task"]
    if task == "summarize":
        return build_summarize_messages(
            example["input"]["article_title"],
            example["input"]["article_text"],
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
        )
    if task == "insight":
        return build_insight_messages(
            example["input"]["summaries"],
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
        )
    raise ValueError(f"알 수 없는 task: {task!r}")


def apply_chat_template(messages: list[dict[str, str]], completion: str, tokenizer: Any) -> str:
    """messages + 정답 completion을 모델별 chat template으로 합쳐 학습 텍스트를 만든다.

    tokenizer.apply_chat_template이 있는 실제 HF tokenizer를 기대한다
    (Qwen3/Gemma4/EXAONE 모두 자체 chat template을 갖고 있어 모델 코드 분기가 필요 없다).
    """
    full_messages = messages + [{"role": "assistant", "content": completion}]
    return tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)


def prompt_only_template(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """completion 없이 prompt까지만 렌더링 — completion-only 마스킹(data.py)에서
    "여기까지는 loss에서 제외" 경계를 찾을 때 사용한다."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
