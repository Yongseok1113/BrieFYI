"""JSONL 입출력 유틸리티. torch 등 무거운 의존성 없이 순수 파이썬으로만 동작한다.

sources/*.py, prepare_data.py, evaluate.py 등 여러 곳에서 공통으로 쓴다.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """JSONL 파일을 한 줄씩 dict로 yield한다. 빈 줄은 건너뛴다."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc


def read_jsonl_list(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], *, mode: str = "w") -> int:
    """records를 JSONL로 저장하고 저장한 줄 수를 반환한다.

    mode="a"로 append 가능 (예: prepare_data.py에서 소스별로 순차 누적할 때).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    return write_jsonl(path, records, mode="a")
