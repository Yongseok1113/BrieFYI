from data_pipeline.llm_client import parse_json_response


def test_parses_plain_json():
    assert parse_json_response('{"canonical": "AI"}') == {"canonical": "AI"}


def test_parses_json_wrapped_in_code_fence():
    text = '```json\n{"canonical": "AI"}\n```'
    assert parse_json_response(text) == {"canonical": "AI"}


def test_parses_json_with_trailing_explanation():
    # Groq llama-3.3-70b-versatile가 코드펜스 없이 JSON 뒤에 부연설명을 덧붙이는
    # 경우를 재현한다 (실제로 normalize 단계에서 "Extra data" json.JSONDecodeError로
    # 실패했던 케이스).
    text = '{"canonical": "AI"}\n\nThis is the best match because...'
    assert parse_json_response(text) == {"canonical": "AI"}


def test_parses_json_with_leading_explanation():
    text = 'Sure, here is the answer:\n{"canonical": "AI"}'
    assert parse_json_response(text) == {"canonical": "AI"}


def test_parses_json_array():
    text = '[{"a": 1}, {"b": 2}]\nnote: two items'
    assert parse_json_response(text) == [{"a": 1}, {"b": 2}]
