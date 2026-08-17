"""공통 학습 예제 스키마.

자체 파이프라인(digests 테이블), AI Hub, Dacon 등 소스가 제각각인 데이터를
sources/*.py가 이 스키마 하나로 변환한다. 이후 단계(data.py, train.py)는
데이터가 어디서 왔는지 신경 쓸 필요가 없다.

design doc 6.2절의 JSON 스키마를 그대로 코드로 옮긴 것.
torch/transformers 등 무거운 의존성이 전혀 없어 GPU 없이도 import/테스트 가능하다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

TaskName = Literal["summarize", "insight", "enrich"]

_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class SchemaError(ValueError):
    """예제가 공통 스키마를 만족하지 못할 때 발생."""


@dataclass
class SummarizeInput:
    article_title: str
    article_text: str
    prompt_template: str = "summarize_v1"

    def to_dict(self) -> dict:
        return {
            "article_title": self.article_title,
            "article_text": self.article_text,
            "prompt_template": self.prompt_template,
        }


@dataclass
class InsightInput:
    summaries: list[dict[str, Any]]
    prompt_template: str = "insight_v1"

    def to_dict(self) -> dict:
        return {"summaries": self.summaries, "prompt_template": self.prompt_template}


@dataclass
class Meta:
    created_at: str
    teacher_model: str = ""
    quality_flag: str = "unverified"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "created_at": self.created_at,
            "teacher_model": self.teacher_model,
            "quality_flag": self.quality_flag,
        }
        d.update(self.extra)
        return d


@dataclass
class Example:
    """학습/평가에서 다루는 단일 예제 (JSONL 한 줄에 대응).

    tools/summarize.py, tools/insight.py가 이미 쓰는 output JSON 스키마를
    그대로 재사용하기 위해 output은 자유 dict로 둔다 — 학습 데이터와
    프로덕션 출력 형식이 항상 일치하도록 하는 것이 목적이다.
    """

    id: str
    task: TaskName
    source: str
    input: dict[str, Any]
    output: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "source": self.source,
            "input": self.input,
            "output": self.output,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        validate_example(d)
        return cls(
            id=d["id"],
            task=d["task"],
            source=d["source"],
            input=d["input"],
            output=d["output"],
            meta=d.get("meta", {}),
        )


_REQUIRED_TOP_LEVEL = ("id", "task", "source", "input", "output")
_VALID_TASKS = ("summarize", "insight", "enrich")

_REQUIRED_INPUT_FIELDS = {
    "summarize": ("article_title", "article_text"),
    "insight": ("summaries",),
    # data_pipeline/의 변형2+변형3 산출물 (기사 1건 -> insights+분류 메타데이터).
    # digests 기반 insight task와 달리 summaries 묶음이 아니라 기사 1건을 입력으로 받는다.
    "enrich": ("article_title", "article_text"),
}

_REQUIRED_OUTPUT_FIELDS = {
    "summarize": ("topic_title", "summary"),
    "insight": ("insights",),
    "enrich": ("insights", "category", "domain", "entity", "event"),
}


def validate_example(d: dict, *, strict_id: bool = False) -> None:
    """예제 dict가 공통 스키마를 만족하는지 검사한다. 위반 시 SchemaError.

    strict_id=True면 id가 uuid4 형식인지까지 검사한다(export 시점에만 사용,
    공개 데이터셋 변환 중간 산출물에는 강제하지 않는다).
    """
    if not isinstance(d, dict):
        raise SchemaError(f"example은 dict여야 함, got {type(d)}")

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in d]
    if missing:
        raise SchemaError(f"필수 필드 누락: {missing}")

    if d["task"] not in _VALID_TASKS:
        raise SchemaError(f"task는 {_VALID_TASKS} 중 하나여야 함, got {d['task']!r}")

    if strict_id and not _ID_RE.match(str(d["id"])):
        raise SchemaError(f"id가 uuid4 형식이 아님: {d['id']!r}")

    if not isinstance(d["source"], str) or not d["source"]:
        raise SchemaError("source는 비어있지 않은 문자열이어야 함")

    task = d["task"]
    input_ = d["input"]
    if not isinstance(input_, dict):
        raise SchemaError("input은 dict여야 함")
    missing_input = [k for k in _REQUIRED_INPUT_FIELDS[task] if k not in input_]
    if missing_input:
        raise SchemaError(f"task={task} input 필수 필드 누락: {missing_input}")

    output = d["output"]
    if not isinstance(output, dict):
        raise SchemaError("output은 dict여야 함")
    missing_output = [k for k in _REQUIRED_OUTPUT_FIELDS[task] if k not in output]
    if missing_output:
        raise SchemaError(f"task={task} output 필수 필드 누락: {missing_output}")

    if task in ("insight", "enrich"):
        insights = output.get("insights")
        if not isinstance(insights, list) or not (3 <= len(insights) <= 5):
            raise SchemaError(
                f"insight 개수는 3~5개여야 함 (evaluate.py 2단계 구조 검증과 동일 기준), got "
                f"{len(insights) if isinstance(insights, list) else type(insights)}"
            )
        for item in insights:
            if not isinstance(item, dict) or "source_url" not in item:
                raise SchemaError("각 insight 항목에는 source_url이 있어야 함")

    if task == "enrich":
        for field_name in ("domain", "entity", "event"):
            if not isinstance(output.get(field_name), list):
                raise SchemaError(f"enrich task의 {field_name}은 배열이어야 함")
        if not isinstance(output.get("category"), str) or not output["category"]:
            raise SchemaError("enrich task의 category는 비어있지 않은 문자열이어야 함")
