import pytest

from summarize_ft.sources.dacon_loader import DaconFormatError, load_csv


def test_load_csv_basic(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "id,title,text,summary\n"
        "1,제목1,본문 내용 1,요약 1\n"
        "2,제목2,본문 내용 2,요약 2\n",
        encoding="utf-8",
    )
    examples = load_csv(csv_path)
    assert len(examples) == 2
    assert examples[0]["task"] == "summarize"
    assert examples[0]["output"]["summary"] == "요약 1"


def test_load_csv_uses_column_aliases(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "id,context,target\n1,본문입니다,요약입니다\n",
        encoding="utf-8",
    )
    examples = load_csv(csv_path)
    assert len(examples) == 1
    assert examples[0]["input"]["article_text"] == "본문입니다"
    assert examples[0]["output"]["summary"] == "요약입니다"


def test_load_csv_skips_empty_rows(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "title,text,summary\n제목,,요약\n제목2,본문,\n제목3,본문3,요약3\n",
        encoding="utf-8",
    )
    examples = load_csv(csv_path)
    assert len(examples) == 1
    assert examples[0]["output"]["summary"] == "요약3"


def test_load_csv_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(DaconFormatError):
        load_csv(csv_path)
