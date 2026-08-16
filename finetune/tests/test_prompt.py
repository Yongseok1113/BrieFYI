from summarize_ft.prompt import (
    build_insight_messages,
    build_messages,
    build_summarize_messages,
    truncate_text,
)


def test_truncate_text_without_tokenizer_short_text_unchanged():
    text = "짧은 텍스트"
    assert truncate_text(text, max_tokens=100) == text


def test_truncate_text_without_tokenizer_long_text_cut():
    text = "가" * 1000
    result = truncate_text(text, max_tokens=10)  # 10 * 2 = 20자로 근사 절단
    assert len(result) == 20


def test_truncate_text_with_tokenizer(fake_tokenizer):
    text = "one two three four five"
    result = truncate_text(text, max_tokens=2, tokenizer=fake_tokenizer)
    assert result == "one two"


def test_build_summarize_messages_has_system_and_user():
    messages = build_summarize_messages("제목", "본문 내용")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "제목" in messages[1]["content"]
    assert "본문 내용" in messages[1]["content"]


def test_build_insight_messages_serializes_summaries():
    summaries = [{"topic_title": "t", "summary": "s", "source_urls": []}]
    messages = build_insight_messages(summaries)
    assert "topic_title" in messages[1]["content"]


def test_build_messages_dispatches_by_task(summarize_example, insight_example):
    summarize_msgs = build_messages(summarize_example)
    insight_msgs = build_messages(insight_example)
    assert "테스트 제목" in summarize_msgs[1]["content"]
    assert len(insight_msgs) == 2
