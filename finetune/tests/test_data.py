import pytest

from summarize_ft.data import IGNORE_INDEX, build_completion_only_example


def test_prompt_portion_is_masked(fake_tokenizer, summarize_example):
    result = build_completion_only_example(summarize_example, fake_tokenizer, max_seq_len=1024)
    input_ids = result["input_ids"]
    labels = result["labels"]

    assert len(input_ids) == len(labels)
    # 앞부분(prompt)은 전부 IGNORE_INDEX여야 함
    assert labels[0] == IGNORE_INDEX
    # completion 부분(뒤쪽)에는 IGNORE_INDEX가 아닌 실제 토큰이 있어야 함
    assert any(label != IGNORE_INDEX for label in labels)


def test_completion_tokens_match_input_ids_tail(fake_tokenizer, summarize_example):
    result = build_completion_only_example(summarize_example, fake_tokenizer, max_seq_len=1024)
    input_ids = result["input_ids"]
    labels = result["labels"]

    # IGNORE_INDEX가 아닌 구간은 input_ids의 같은 위치와 정확히 일치해야 한다
    for token_id, label in zip(input_ids, labels):
        if label != IGNORE_INDEX:
            assert token_id == label


def test_max_seq_len_truncates(fake_tokenizer, summarize_example):
    result = build_completion_only_example(summarize_example, fake_tokenizer, max_seq_len=5)
    assert len(result["input_ids"]) <= 5
    assert len(result["labels"]) <= 5


def test_dynamic_padding_collator():
    torch = pytest.importorskip("torch")
    from summarize_ft.data import DynamicPaddingCollator

    collator = DynamicPaddingCollator(pad_token_id=0)
    batch = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, IGNORE_INDEX, 3]},
        {"input_ids": [1, 2], "labels": [IGNORE_INDEX, 2]},
    ]
    out = collator(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"][1][2].item() == IGNORE_INDEX  # 패딩된 label 위치
