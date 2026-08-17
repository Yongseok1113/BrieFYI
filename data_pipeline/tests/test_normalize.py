import data_pipeline.normalize as normalize_module
from data_pipeline.normalize import normalize_value

ENTRIES = [
    {"canonical_value": "삼성전자", "aliases": ["삼성", "Samsung Electronics"]},
    {"canonical_value": "NVIDIA", "aliases": ["엔비디아"]},
]


def test_exact_match_on_canonical():
    result, method = normalize_value("삼성전자", ENTRIES, fuzzy_threshold=0.85)
    assert result == "삼성전자"
    assert method == "exact"


def test_exact_match_on_alias():
    result, method = normalize_value("엔비디아", ENTRIES, fuzzy_threshold=0.85)
    assert result == "NVIDIA"
    assert method == "exact"


def test_fuzzy_match_close_typo():
    # "삼성전자 " (트레일링 스페이스)는 정확히 일치하진 않지만 fuzzy로는 잡혀야 함
    result, method = normalize_value("삼성전자 ", ENTRIES, fuzzy_threshold=0.85)
    assert result == "삼성전자"
    assert method == "fuzzy"


def test_llm_fallback_used_when_no_match(monkeypatch):
    def fake_call_llm(system, user, **kwargs):
        return '{"canonical": "새로운값"}'

    monkeypatch.setattr(normalize_module, "call_llm", fake_call_llm)

    result, method = normalize_value("완전히 다른 개념", ENTRIES, fuzzy_threshold=0.99)
    assert result == "새로운값"
    assert method == "llm"


def test_normalize_list_reports_overall_method():
    from data_pipeline.normalize import _normalize_list

    values, method = _normalize_list(["삼성전자", "NVIDIA"], ENTRIES, fuzzy_threshold=0.85)
    assert values == ["삼성전자", "NVIDIA"]
    assert method == "exact"


def test_normalize_list_empty_returns_exact():
    from data_pipeline.normalize import _normalize_list

    values, method = _normalize_list([], ENTRIES, fuzzy_threshold=0.85)
    assert values == []
    assert method == "exact"
