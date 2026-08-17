from data_pipeline.keywords import extract_keywords


def test_empty_text_returns_empty_list():
    assert extract_keywords("") == []
    assert extract_keywords("   ") == []


def test_fallback_extracts_frequent_tokens():
    text = "삼성전자 삼성전자 반도체 투자 반도체 발표 삼성전자"
    result = extract_keywords(text, top_n=2, use_keybert=False)
    assert result[0] == "삼성전자"
    assert "반도체" in result


def test_fallback_respects_top_n():
    text = "가나다 가나다 라마바 라마바 사아자 사아자 차카타 차카타"
    result = extract_keywords(text, top_n=2, use_keybert=False)
    assert len(result) == 2


def test_fallback_ignores_short_tokens_and_stopwords():
    text = "이 것 은 AI 관련 기사 이다"
    result = extract_keywords(text, top_n=10, use_keybert=False)
    assert "이" not in result
    assert "것" not in result
