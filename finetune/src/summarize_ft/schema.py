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
        if not isinstance(insights, list):
            raise SchemaError(f"insights는 배열이어야 함, got {type(insights)}")
        # make_train_data의 환각 방지 calibration 예제(단발성 사실 기사)는 억지로 인사이트를
        # 채우지 않고 insights=[]로 두는 대신 output.no_strong_insight=True를 명시해야 한다.
        # 그 표시 없이 0개면 아래 기준(2~5개) 위반으로 취급한다.
        #
        # 하한을 3이 아니라 2로 잡은 이유: tools/insight.py(digests 기반) 프롬프트는 자체적으로
        # "3~5개"를 요청하므로 실제로는 항상 3개 이상 나온다 -- 그쪽 동작은 이 하한과 무관하다.
        # 반면 make_train_data(cluster_export.py)의 claude_prompt는 "최소 2종 이상 시도, 억지로
        # 6종 다 채우지 않음"이라고 명시한다. 하한을 3으로 두면 클러스터 기반 골드셋 생성 시
        # 근거 없는 3번째 인사이트를 억지로 채워야 해서, 오히려 이 스키마가 막으려는 문제
        # (근거 부족한 인사이트 강제 생성)를 유발한다. 그래서 2~5로 완화했다.
        if len(insights) == 0:
            if not output.get("no_strong_insight"):
                raise SchemaError(
                    "insights가 0개면 output.no_strong_insight=True여야 함 "
                    "(그 외의 경우 2~5개, evaluate.py 2단계 구조 검증과 동일 기준)"
                )
        elif not (2 <= len(insights) <= 5):
            raise SchemaError(
                f"insight 개수는 0개(no_strong_insight=True) 또는 2~5개여야 함, got "
                f"{len(insights)}"
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
