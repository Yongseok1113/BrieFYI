"""AI Hub "문서요약 텍스트" 데이터셋 파서 (design doc 3.2절).

원문: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=97 (승인 후 다운로드 필요)

AI Hub 문서요약 텍스트는 신문기사/기고문/잡지/판결문 각각 JSON 파일로 오고,
문서 하나가 대략 아래 형태다 (필드명은 배포본마다 조금씩 달라질 수 있어
`--sample-check`로 실제 파일 1개를 먼저 찍어보고 확정할 것을 권장한다):

    {
      "Meta(Refine)": {"passage": "...", "publisher": "..."},
      "Annotation": {
        "text": [{"index": 0, "sentence": "..."}, ...],
        "summary1": "생성(추상) 요약문",
        "summary2": "생성(추상) 요약문 2 (있는 경우)",
        "summary_extractive": [0, 3, 5]
      }
    }

이 로더는 위 구조를 가정해 "abstractive summary1"을 정답으로 채택한다
(요약 형식·정보 압축 능력을 배우는 base fine-tune 용도이므로 추상 요약을 우선).
실제 배포본 필드명이 다르면 _extract_text/_extract_summary 두 함수만 고치면 된다.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class AihubFormatError(ValueError):
    """예상한 필드가 없을 때 발생 — 실제 배포본과 필드명이 다르다는 신호."""


def _extract_text(doc: dict) -> str:
    annotation = doc.get("Annotation", {})
    sentences = annotation.get("text")
    if isinstance(sentences, list):
        return "\n".join(s.get("sentence", "") for s in sentences if isinstance(s, dict))
    meta = doc.get("Meta(Refine)", {})
    if "passage" in meta:
        return meta["passage"]
    raise AihubFormatError("Annotation.text 또는 Meta(Refine).passage를 찾지 못함")


def _extract_summary(doc: dict) -> str:
    annotation = doc.get("Annotation", {})
    for key in ("summary1", "summary2", "summary"):
        if annotation.get(key):
            return annotation[key]
    raise AihubFormatError("Annotation.summary1/summary2/summary를 찾지 못함")


def _extract_title(doc: dict) -> str:
    meta = doc.get("Meta(Refine)", {})
    return meta.get("title") or meta.get("publisher") or ""


def load_file(path: str | Path) -> list[dict[str, Any]]:
    """AI Hub JSON 파일 1개(문서 여러 건을 담은 배열 또는 단건)를 공통 스키마로 변환한다."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = raw if isinstance(raw, list) else [raw]

    examples = []
    for doc in docs:
        try:
            article_text = _extract_text(doc)
            summary = _extract_summary(doc)
        except AihubFormatError:
            continue  # 필드가 없는 손상/예외 문서는 건너뛴다 (3.3절 데이터 정제 원칙)

        examples.append(
            {
                "id": str(uuid.uuid4()),
                "task": "summarize",
                "source": "aihub_document_summary",
                "input": {
                    "article_title": _extract_title(doc),
                    "article_text": article_text,
                    "prompt_template": "summarize_v1",
                },
                "output": {
                    "topic_title": _extract_title(doc),
                    "summary": summary,
                    "source_urls": [],  # 공개 데이터셋은 원본 URL이 없는 경우가 많음
                },
                "meta": {
                    "created_at": "",
                    "teacher_model": "",  # 사람이 작성한 정답 요약, teacher 모델 없음
                    "quality_flag": "verified",  # AI Hub는 사람 검수를 거친 데이터라 verified로 시작
                },
            }
        )
    return examples


def load_dir(dir_path: str | Path, pattern: str = "*.json") -> list[dict[str, Any]]:
    """디렉터리 내 모든 AI Hub JSON 파일을 순회하며 합친다."""
    dir_path = Path(dir_path)
    examples: list[dict[str, Any]] = []
    for file_path in sorted(dir_path.glob(pattern)):
        examples.extend(load_file(file_path))
    return examples


if __name__ == "__main__":
    import argparse

    from ..jsonl import write_jsonl

    parser = argparse.ArgumentParser(description="AI Hub 문서요약 텍스트 -> 공통 스키마 JSONL")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", default="finetune/data/processed/aihub.jsonl")
    parser.add_argument(
        "--sample-check", action="store_true",
        help="첫 파일 1건만 파싱해 필드가 예상과 맞는지 출력하고 종료"
    )
    args = parser.parse_args()

    if args.sample_check:
        first = sorted(Path(args.in_dir).glob("*.json"))[0]
        result = load_file(first)
        print(json.dumps(result[:1], ensure_ascii=False, indent=2))
    else:
        examples = load_dir(args.in_dir)
        n = write_jsonl(args.out, examples)
        print(f"{n}건 export 완료 -> {args.out}")
