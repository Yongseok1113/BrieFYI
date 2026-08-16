"""Dacon 뉴스 요약 경진대회 데이터 파서 (design doc 3.2절).

Dacon 대회 데이터는 대회마다 컬럼명이 다르지만, 뉴스 요약류는 보통
`train.csv`에 `id, title, text(또는 context), summary` 형태의 컬럼을 갖는다.
실제 대회 데이터를 받으면 CSV 헤더를 확인해 _COLUMN_ALIASES에 실제 컬럼명을
추가하기만 하면 되도록 별칭(alias) 매핑 방식으로 짰다 — 대회마다 새 로더를
또 만들 필요가 없게 하기 위함.
"""
from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Any

# 실제 대회 데이터의 헤더가 다르면 여기 별칭만 추가하면 된다.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "text": ["text", "context", "article", "본문", "content"],
    "title": ["title", "제목"],
    "summary": ["summary", "abstractive", "요약", "target"],
}


class DaconFormatError(ValueError):
    pass


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        match = next((f for f in fieldnames if f.strip().lower() in [a.lower() for a in aliases]), None)
        if match:
            resolved[canonical] = match
    if "text" not in resolved or "summary" not in resolved:
        raise DaconFormatError(
            f"text/summary 컬럼을 찾지 못함. 실제 헤더: {fieldnames} — _COLUMN_ALIASES에 별칭 추가 필요"
        )
    return resolved


def load_csv(path: str | Path, *, source_name: str = "dacon_news_summary") -> list[dict[str, Any]]:
    path = Path(path)
    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = _resolve_columns(reader.fieldnames or [])

        for row in reader:
            text = (row.get(columns["text"]) or "").strip()
            summary = (row.get(columns["summary"]) or "").strip()
            if not text or not summary:
                continue  # 3.3절: 빈 값은 학습 데이터로 채택하지 않는다

            title = (row.get(columns.get("title", "")) or "").strip() if "title" in columns else ""

            examples.append(
                {
                    "id": str(uuid.uuid4()),
                    "task": "summarize",
                    "source": source_name,
                    "input": {
                        "article_title": title,
                        "article_text": text,
                        "prompt_template": "summarize_v1",
                    },
                    "output": {
                        "topic_title": title,
                        "summary": summary,
                        "source_urls": [],
                    },
                    "meta": {
                        "created_at": "",
                        "teacher_model": "",
                        "quality_flag": "verified",
                    },
                }
            )
    return examples


if __name__ == "__main__":
    import argparse

    from ..jsonl import write_jsonl

    parser = argparse.ArgumentParser(description="Dacon 뉴스 요약 CSV -> 공통 스키마 JSONL")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="finetune/data/processed/dacon.jsonl")
    args = parser.parse_args()

    examples = load_csv(args.csv)
    n = write_jsonl(args.out, examples)
    print(f"{n}건 export 완료 -> {args.out}")
