from summarize_ft.config import Config
from summarize_ft.registry import best_run, log_run, read_runs


def test_log_run_appends(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    cfg = Config(base_model="Qwen/Qwen3-8B-Instruct")

    log_run(cfg, "hash1", "train", {}, registry_path=str(registry_path))
    log_run(cfg, "hash1", "evaluate", {"rougeL_f1": 0.4}, registry_path=str(registry_path))

    runs = read_runs(str(registry_path))
    assert len(runs) == 2
    assert runs[0]["stage"] == "train"
    assert runs[1]["metrics"]["rougeL_f1"] == 0.4


def test_read_runs_missing_file_returns_empty(tmp_path):
    assert read_runs(str(tmp_path / "nope.jsonl")) == []


def test_best_run_picks_highest_metric(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    cfg = Config(base_model="x")

    log_run(cfg, "h1", "evaluate", {"rougeL_f1": 0.3}, registry_path=str(registry_path))
    log_run(cfg, "h2", "evaluate", {"rougeL_f1": 0.6}, registry_path=str(registry_path))
    log_run(cfg, "h3", "train", {}, registry_path=str(registry_path))  # stage 다름 -> 제외돼야 함

    best = best_run(str(registry_path))
    assert best["config_hash"] == "h2"


def test_best_run_no_matching_returns_none(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    assert best_run(str(registry_path)) is None
