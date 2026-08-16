"""completion-only 마스킹 + 동적 패딩 콜레이터 + Dataset 빌드.

프롬프트 부분은 loss 계산에서 제외(label=-100)하고 completion(정답 요약/인사이트
JSON) 부분만 학습 신호로 쓴다. 이걸 안 하면 모델이 "프롬프트를 그대로 베끼는"
학습을 하게 되어 요약 품질이 잘 안 붙는다.

torch/datasets는 실제 학습 시에만 필요하므로 지연 import한다 — 이 모듈을
import하는 것 자체는 torch 없이도 가능해야 tests/에서 로직만 가볍게 검증할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompt import apply_chat_template, build_messages, prompt_only_template

IGNORE_INDEX = -100


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - GPU 환경에서만 실행됨
        raise ImportError(
            "data.py의 이 기능은 torch가 필요합니다. `pip install -r finetune/requirements.txt`"
        ) from exc
    return torch


def build_completion_only_example(example: dict[str, Any], tokenizer: Any,
                                   max_seq_len: int = 1024) -> dict[str, list[int]]:
    """예제 하나를 (input_ids, labels) 쌍으로 변환한다.

    labels는 prompt 구간을 IGNORE_INDEX(-100)로 마스킹하고 completion 구간만
    실제 토큰 id를 채운다. transformers는 -100을 CrossEntropyLoss에서 자동으로
    무시하므로 loss_mask를 별도로 넘길 필요가 없다.
    """
    import json

    messages = build_messages(example)
    completion = json.dumps(example["output"], ensure_ascii=False)

    prompt_text = prompt_only_template(messages, tokenizer)
    full_text = apply_chat_template(messages, completion, tokenizer)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * prompt_len + list(full_ids[prompt_len:])
    labels = labels[: len(full_ids)]

    return {"input_ids": full_ids, "labels": labels}


@dataclass
class DynamicPaddingCollator:
    """배치 내 최대 길이에 맞춰서만 패딩한다 (고정 max_seq_len 패딩보다 학습이 빠르다)."""

    pad_token_id: int
    label_pad_id: int = IGNORE_INDEX

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        torch = _require_torch()

        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []

        for item in batch:
            ids = item["input_ids"]
            lbl = item["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbl + [self.label_pad_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(jsonl_path: str, tokenizer: Any, max_seq_len: int = 1024):
    """JSONL 경로 -> HF datasets.Dataset. train.py의 SFTTrainer에 그대로 넘긴다."""
    from datasets import Dataset

    from .jsonl import read_jsonl_list
    from .schema import validate_example

    records = read_jsonl_list(jsonl_path)
    processed = []
    for record in records:
        validate_example(record)
        processed.append(build_completion_only_example(record, tokenizer, max_seq_len=max_seq_len))
    return Dataset.from_list(processed)
