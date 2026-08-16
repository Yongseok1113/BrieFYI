"""실험 레지스트리 — 학습/평가 실행마다 config 해시, 데이터 버전, 평가 점수를 기록한다.

기본은 JSON Lines 파일(runs/registry.jsonl)에 append하고, 필요해지면(예: 여러
사람이 동시에 실험할 때) Postgres `finetune_runs` 테이블로 옮길 수 있게 스키마를
단순하게 유지한다. docs/WORKLOG.md는 이 로그를 사람이 읽기 좋게 요약하는 용도로
남겨두되, 실제 비교·검색은 이 파일의 구조화된 로그를 기준으로 한다 (design doc 6.6절).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import Config

DEFAULT_REGISTRY_PATH = "runs/registry.jsonl"


def log_run(config: Config, config_hash: str, stage: str, metrics: dict[str, Any],
            *, registry_path: str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """실행 1건을 registry에 append하고 기록한 row를 반환한다."""
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,  # "train" | "evaluate"
        "config_hash": config_hash,
        "base_model": config.base_model,
        "task": config.task,
        "data_version": config.data.data_version,
        "output_dir": config.output_dir,
        "metrics": metrics,
    }

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
    return row


def read_runs(registry_path: str = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def best_run(registry_path: str = DEFAULT_REGISTRY_PATH, *, metric: str = "rougeL_f1") -> dict[str, Any] | None:
    """metric 기준으로 evaluate 단계 실행 중 가장 좋은 run을 반환한다 (없으면 None)."""
    runs = [r for r in read_runs(registry_path) if r["stage"] == "evaluate" and metric in r.get("metrics", {})]
    if not runs:
        return None
    return max(runs, key=lambda r: r["metrics"][metric])
