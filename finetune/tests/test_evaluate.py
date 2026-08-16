from summarize_ft.evaluate import stage2_structural_validation, stage4_export_pairwise


def test_stage2_all_pass(summarize_example, insight_example):
    report = stage2_structural_validation([summarize_example, insight_example])
    assert report.metrics["pass_rate"] == 1.0
    assert report.metrics["passed"] == 2


def test_stage2_reports_failures(summarize_example):
    broken = dict(summarize_example)
    broken["output"] = {}  # 필수 필드 누락 -> 실패해야 함
    report = stage2_structural_validation([summarize_example, broken])
    assert report.metrics["passed"] == 1
    assert report.metrics["total"] == 2
    assert report.metrics["pass_rate"] == 0.5
    assert report.metrics["top_failures"]


def test_stage2_empty_input():
    report = stage2_structural_validation([])
    assert report.metrics["pass_rate"] == 0.0
    assert report.metrics["total"] == 0


def test_stage4_export_pairwise_writes_file(tmp_path, summarize_example):
    out_path = tmp_path / "pairwise.jsonl"
    report = stage4_export_pairwise(
        [summarize_example], ["baseline 요약"], ["candidate 요약"], str(out_path)
    )
    assert report.metrics["exported"] == 1
    assert out_path.exists()

    from summarize_ft.jsonl import read_jsonl_list

    rows = read_jsonl_list(out_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["winner"] is None
    assert set(row["label_map"].values()) == {"baseline", "candidate"}
    assert {row["option_a"], row["option_b"]} == {"baseline 요약", "candidate 요약"}
