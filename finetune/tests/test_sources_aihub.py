import json

from summarize_ft.sources.aihub_loader import load_file


def test_load_file_extracts_sentences_and_summary(tmp_path):
    doc = {
        "Meta(Refine)": {"title": "제목", "publisher": "매체"},
        "Annotation": {
            "text": [{"index": 0, "sentence": "첫 문장."}, {"index": 1, "sentence": "둘째 문장."}],
            "summary1": "요약 문장.",
        },
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")

    examples = load_file(path)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["task"] == "summarize"
    assert "첫 문장." in ex["input"]["article_text"]
    assert ex["output"]["summary"] == "요약 문장."
    assert ex["meta"]["quality_flag"] == "verified"


def test_load_file_skips_malformed_doc(tmp_path):
    good = {
        "Meta(Refine)": {"title": "제목"},
        "Annotation": {"text": [{"sentence": "본문"}], "summary1": "요약"},
    }
    bad = {"Meta(Refine)": {}, "Annotation": {}}  # text/summary 둘 다 없음
    path = tmp_path / "doc.json"
    path.write_text(json.dumps([good, bad], ensure_ascii=False), encoding="utf-8")

    examples = load_file(path)
    assert len(examples) == 1  # bad는 건너뜀


def test_load_file_single_doc_not_list(tmp_path):
    doc = {
        "Meta(Refine)": {"title": "단건"},
        "Annotation": {"text": [{"sentence": "본문"}], "summary1": "요약"},
    }
    path = tmp_path / "single.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    examples = load_file(path)
    assert len(examples) == 1
