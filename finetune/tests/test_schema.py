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
    insight_example["output"]["insights"] = insight_example["output"]["insights"][:1]  # 1개 -> 2~5 위반
    with pytest.raises(SchemaError):
        validate_example(insight_example)


def test_insight_count_two_passes(insight_example):
    # make_train_data(cluster_export.py)는 "최소 2종 이상 시도"만 요구하므로 2개도 유효해야 함
    insight_example["output"]["insights"] = insight_example["output"]["insights"][:2]
    validate_example(insight_example)  # 예외 없이 통과해야 함


def test_insight_missing_source_url_raises(insight_example):
    del insight_example["output"]["insights"][0]["source_url"]
    with pytest.raises(SchemaError):
        validate_example(insight_example)


def test_insight_zero_with_no_strong_insight_flag_passes(insight_example):
    insight_example["output"]["insights"] = []
    insight_example["output"]["no_strong_insight"] = True
    validate_example(insight_example)  # 예외 없이 통과해야 함 (make_train_data calibration 예제)


def test_insight_zero_without_flag_raises(insight_example):
    insight_example["output"]["insights"] = []
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


def test_valid_enrich_example_passes(enrich_example):
    validate_example(enrich_example)


def test_enrich_missing_category_raises(enrich_example):
    del enrich_example["output"]["category"]
    with pytest.raises(SchemaError):
        validate_example(enrich_example)


def test_enrich_domain_must_be_list(enrich_example):
    enrich_example["output"]["domain"] = "반도체"  # 배열이어야 하는데 문자열
    with pytest.raises(SchemaError):
        validate_example(enrich_example)


def test_enrich_reuses_insight_count_check(enrich_example):
    enrich_example["output"]["insights"] = enrich_example["output"]["insights"][:1]
    with pytest.raises(SchemaError):
        validate_example(enrich_example)
