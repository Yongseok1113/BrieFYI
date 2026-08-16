import pytest

from summarize_ft.schema import Example, SchemaError, validate_example


def test_valid_summarize_example_passes(summarize_example):
    validate_example(summarize_example)  # 예외 없이 통과해야 함


def test_valid_insight_example_passes(insight_example):
    validate_example(insight_example)


def test_missing_required_field_raises(summarize_example):
    del summarize_example["output"]
    with pytest.raises(SchemaError):
        validate_example(summarize_example)


def test_invalid_task_raises(summarize_example):
    summarize_example["task"] = "translate"
    with pytest.raises(SchemaError):
        validate_example(summarize_example)


def test_insight_count_out_of_range_raises(insight_example):
    insight_example["output"]["insights"] = insight_example["output"]["insights"][:2]  # 2개 -> 3~5 위반
    with pytest.raises(SchemaError):
        validate_example(insight_example)


def test_insight_missing_source_url_raises(insight_example):
    del insight_example["output"]["insights"][0]["source_url"]
    with pytest.raises(SchemaError):
        validate_example(insight_example)


def test_summarize_missing_input_field_raises(summarize_example):
    del summarize_example["input"]["article_text"]
    with pytest.raises(SchemaError):
        validate_example(summarize_example)


def test_strict_id_rejects_non_uuid(summarize_example):
    summarize_example["id"] = "not-a-uuid"
    with pytest.raises(SchemaError):
        validate_example(summarize_example, strict_id=True)


def test_example_roundtrip(summarize_example):
    ex = Example.from_dict(summarize_example)
    assert ex.to_dict() == summarize_example


def test_non_dict_raises():
    with pytest.raises(SchemaError):
        validate_example("not a dict")  # type: ignore[arg-type]
