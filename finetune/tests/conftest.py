"""공통 픽스처. 모델 다운로드 없이 순수 파이썬 로직만 검증하는 것이 목표라
tokenizer는 진짜 HF tokenizer 대신 최소 duck-type FakeTokenizer를 쓴다.
"""
from __future__ import annotations

import pytest


class FakeTokenizer:
    """공백 단위 토큰화 + 아주 단순한 chat template. torch/transformers 불필요.

    prompt.py, data.py가 요구하는 encode/decode/apply_chat_template 인터페이스만
    구현한다.
    """

    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._token_id(tok) for tok in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False,
                             add_generation_prompt: bool = False) -> str:
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def summarize_example() -> dict:
    return {
        "id": "11111111-1111-4111-8111-111111111111",
        "task": "summarize",
        "source": "test",
        "input": {
            "article_title": "테스트 제목",
            "article_text": "테스트 본문 내용입니다 여러 단어로 구성됨",
            "prompt_template": "summarize_v1",
        },
        "output": {
            "topic_title": "테스트 제목",
            "summary": "테스트 요약",
            "source_urls": ["https://example.com/1"],
        },
        "meta": {"created_at": "2026-01-01T00:00:00Z", "teacher_model": "claude", "quality_flag": "verified"},
    }


@pytest.fixture
def insight_example() -> dict:
    return {
        "id": "22222222-2222-4222-8222-222222222222",
        "task": "insight",
        "source": "test",
        "input": {"summaries": [{"topic_title": "t", "summary": "s", "source_urls": []}]},
        "output": {
            "insights": [
                {"text": "인사이트1", "source_url": "https://example.com/1"},
                {"text": "인사이트2", "source_url": "https://example.com/2"},
                {"text": "인사이트3", "source_url": "https://example.com/3"},
            ],
            "business_implication": "시사점 문단",
        },
        "meta": {"created_at": "2026-01-01T00:00:00Z", "teacher_model": "claude", "quality_flag": "verified"},
    }
