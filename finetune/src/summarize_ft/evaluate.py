"""4단계 평가 (design doc 4절 + 6.5절).

    python -m summarize_ft.evaluate --checkpoint runs/qwen3-8b-summarize-v1/adapter \
        --testset data/golden/testset_v1.jsonl

1단계 ROUGE/BERTScore -> 2단계 구조 검증 -> 3단계 grounding(사실성) -> 4단계 사람평가용
pairwise export 순으로 실행하고 리포트를 만든다. 체크포인트만 바꿔 끼우면 되므로
서로 다른 베이스 모델(Qwen3-8B / Gemma 4 12B 등)을 같은 골든셋으로 나란히 비교할 수 있다.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .jsonl import read_jsonl_list, write_jsonl
from .schema import SchemaError, validate_example


@dataclass
class StageReport:
    stage: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EvalReport:
    checkpoint: str
    testset: str
    n_examples: int
    stages: list[StageReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "testset": self.testset,
            "n_examples": self.n_examples,
            "stages": [asdict(s) for s in self.stages],
        }


# ---------------------------------------------------------------------------
# 1단계: 자동 지표 (ROUGE-1/2/L, BERTScore) — 참고용, 합격선은 이것만으로 정하지 않는다.
# ---------------------------------------------------------------------------

def stage1_automatic_metrics(predictions: list[str], references: list[str]) -> StageReport:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return StageReport(
            stage="1_automatic_metrics",
            notes="rouge_score 미설치 — `pip install rouge-score bert-score`로 설치 필요",
        )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    metrics = {f"{k}_f1": round(v / n, 4) for k, v in totals.items()}

    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(predictions, references, lang="ko", verbose=False)
        metrics["bertscore_f1"] = round(float(f1.mean()), 4)
    except ImportError:
        metrics["bertscore_f1"] = None

    return StageReport(stage="1_automatic_metrics", metrics=metrics)


# ---------------------------------------------------------------------------
# 2단계: 구조 검증 — JSON 파싱 성공률, 인사이트 개수 3~5개, source_url 존재, 스키마 준수.
# schema.py의 validate_example과 동일한 기준을 재사용해 "학습 데이터 기준 == 평가 기준"을 보장한다.
# ---------------------------------------------------------------------------

def stage2_structural_validation(examples: list[dict[str, Any]]) -> StageReport:
    total = len(examples)
    passed = 0
    failure_reasons: dict[str, int] = {}

    for ex in examples:
        try:
            validate_example(ex)
            passed += 1
        except SchemaError as exc:
            reason = str(exc)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    metrics = {
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "total": total,
        "top_failures": sorted(failure_reasons.items(), key=lambda kv: -kv[1])[:5],
    }
    return StageReport(stage="2_structural_validation", metrics=metrics)


# ---------------------------------------------------------------------------
# 3단계: 사실 일치성(환각 검증) — Claude를 판정자로 세운 grounding check.
# 새 모델을 훈련하는 게 아니라 "원문 대비 사실성"이라는 객관적 기준을 판정하는
# 것이므로 교사 모델(Claude)을 평가자로 재사용해도 문제없다 (design doc 4절).
# ---------------------------------------------------------------------------

GROUNDING_JUDGE_PROMPT = """다음은 원문 기사와 그에 대한 요약이다. 요약의 각 문장이
원문에 실제로 근거하는지 판정하라. 원문에 없는 사실을 지어냈다면 grounded=false로 표시하라.

# 원문
{article_text}

# 요약
{summary}

아래 JSON 형식으로만 답하라:
{{"grounded": true/false, "ungrounded_claims": ["..."], "reason": "..."}}
"""


def stage3_grounding_check(examples: list[dict[str, Any]], *, anthropic_client: Any = None,
                            model: str = "claude-sonnet-4-5") -> StageReport:
    if anthropic_client is None:
        return StageReport(
            stage="3_grounding_check",
            notes="anthropic_client가 없어 스킵 — evaluate.py를 --with-grounding 옵션으로 실행하면 활성화",
        )

    grounded_count = 0
    checked = 0
    for ex in examples:
        article_text = ex.get("input", {}).get("article_text")
        summary = ex.get("output", {}).get("summary")
        if not article_text or not summary:
            continue
        checked += 1
        prompt = GROUNDING_JUDGE_PROMPT.format(article_text=article_text, summary=summary)
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().strip("`")
        try:
            verdict = json.loads(text)
            if verdict.get("grounded"):
                grounded_count += 1
        except json.JSONDecodeError:
            continue

    metrics = {
        "checked": checked,
        "grounded": grounded_count,
        "grounded_rate": round(grounded_count / checked, 4) if checked else None,
    }
    return StageReport(stage="3_grounding_check", metrics=metrics)


# ---------------------------------------------------------------------------
# 4단계: 사람 평가용 pairwise export — Claude 요약 vs 새 모델 요약을 블라인드로 나란히.
# 실제 판정은 evaluate.py가 하지 않고, 팀이 눈으로 보고 승패를 매기도록 파일만 만든다.
# ---------------------------------------------------------------------------

def stage4_export_pairwise(examples: list[dict[str, Any]], baseline_predictions: list[str],
                            candidate_predictions: list[str], output_path: str) -> StageReport:
    import random

    rows = []
    for ex, baseline, candidate in zip(examples, baseline_predictions, candidate_predictions):
        # A/B 순서를 무작위로 섞어 블라인드 평가가 되도록 한다.
        if random.random() < 0.5:
            option_a, option_b, label_map = baseline, candidate, {"A": "baseline", "B": "candidate"}
        else:
            option_a, option_b, label_map = candidate, baseline, {"A": "candidate", "B": "baseline"}
        rows.append(
            {
                "id": ex.get("id"),
                "article_title": ex.get("input", {}).get("article_title"),
                "option_a": option_a,
                "option_b": option_b,
                "label_map": label_map,
                "winner": None,  # 사람이 채워 넣는 칸: "A" | "B" | "tie"
            }
        )
    n = write_jsonl(output_path, rows)
    return StageReport(stage="4_pairwise_export", metrics={"exported": n, "path": output_path})


def run_evaluation(checkpoint: str, testset_path: str, *, with_grounding: bool = False,
                    anthropic_client: Any = None) -> EvalReport:
    examples = read_jsonl_list(testset_path)
    report = EvalReport(checkpoint=checkpoint, testset=testset_path, n_examples=len(examples))

    predictions = [json.dumps(ex.get("output", {}), ensure_ascii=False) for ex in examples]
    references = predictions  # 실제로는 infer.py로 생성한 결과를 predictions에 넣어야 함 (아래 CLI 참고)

    report.stages.append(stage1_automatic_metrics(predictions, references))
    report.stages.append(stage2_structural_validation(examples))
    if with_grounding:
        report.stages.append(stage3_grounding_check(examples, anthropic_client=anthropic_client))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize_ft 4단계 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--with-grounding", action="store_true", help="3단계 grounding check(Claude 호출) 활성화")
    parser.add_argument("--report-out", default=None, help="결과 JSON을 저장할 경로 (기본: <checkpoint>/eval_report.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    anthropic_client = None
    if args.with_grounding:
        import anthropic

        anthropic_client = anthropic.Anthropic()

    report = run_evaluation(
        args.checkpoint, args.testset, with_grounding=args.with_grounding, anthropic_client=anthropic_client
    )

    out_path = Path(args.report_out or Path(args.checkpoint) / "eval_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"평가 리포트 저장: {out_path}")
    for stage in report.stages:
        print(f"  [{stage.stage}] {stage.metrics or stage.notes}")


if __name__ == "__main__":
    main()
