import pytest

from summarize_ft.jsonl import append_jsonl, read_jsonl_list, write_jsonl


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "out.jsonl"
    records = [{"a": 1}, {"b": "한글"}]
    n = write_jsonl(path, records)
    assert n == 2
    assert read_jsonl_list(path) == records


def test_write_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl_list(path) == [{"a": 1}, {"b": 2}]


def test_append_adds_to_existing(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [{"a": 1}])
    append_jsonl(path, [{"a": 2}])
    assert read_jsonl_list(path) == [{"a": 1}, {"a": 2}]


def test_malformed_line_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl_list(path)


def test_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.jsonl"
    write_jsonl(path, [{"a": 1}])
    assert path.exists()
