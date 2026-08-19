import pytest

from summarize_ft.infer import GenerationError, _parse_json_output


def test_plain_json_parses():
    assert _parse_json_output('{"a": 1}') == {"a": 1}


def test_code_fence_wrapped_json_parses():
    assert _parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}


def test_think_block_before_json_is_stripped():
    # Qwen3 등 하이브리드 추론 모델이 실제로 내는 형태 (스모크 테스트에서 관찰됨)
    text = '<think>\n\n</think>\n\n{"facts": ["x"], "insights": [], "no_strong_insight": true}'
    assert _parse_json_output(text) == {
        "facts": ["x"],
        "insights": [],
        "no_strong_insight": True,
    }


def test_think_block_with_content_is_stripped():
    text = '<think>이 기사는 단발성 사실이다.</think>\n{"a": 1}'
    assert _parse_json_output(text) == {"a": 1}


def test_unclosed_think_block_raises():
    with pytest.raises(GenerationError):
        _parse_json_output('<think>끝나지 않은 추론 {"a": 1}')


def test_invalid_json_raises():
    with pytest.raises(GenerationError):
        _parse_json_output("이건 JSON이 아닙니다")
