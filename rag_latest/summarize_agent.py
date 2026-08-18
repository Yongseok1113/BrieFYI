"""검색 결과 요약 -> grounding 검증 -> LLM judge 판정 -> 재시도 루프.

search_news()가 반환한 chunk 목록을 받아 요약을 생성하고, 그 요약이 실제로 chunk에
근거하는지 검증한 뒤, 종합 점수를 매겨 임계치 미달이면 재생성한다. 매 시도는
db.save_agent_run()으로 즉시 기록한다. max_attempts에 도달하면 예외를 던지지 않고
마지막 시도를 passed=False로 반환한다 — 실패도 정상적인 반환 경로다.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from config import config

from .db import save_agent_run
from .llm_client import LLMError, call_llm, parse_json_response
from .prompts import (
    GROUNDING_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    build_grounding_user_prompt,
    build_judge_user_prompt,
    build_summarize_user_prompt,
)


@dataclass
class AttemptResult:
    attempt_number: int
    summary: str
    citations: list[dict]
    grounding_passed: bool
    grounding_issues: list[dict]
    judge_score: float
    judge_reasoning: str
    passed_threshold: bool


@dataclass
class SummarizeResult:
    run_id: str
    query: str
    attempts: list[AttemptResult]

    @property
    def final(self) -> AttemptResult:
        return self.attempts[-1]

    @property
    def passed(self) -> bool:
        return bool(self.attempts) and self.final.passed_threshold


def _run_one_attempt(query: str, chunks: list[dict], provider: str, score_threshold: float) -> AttemptResult | None:
    """파싱 실패 시 None(호출부가 실패로 기록하고 다음 시도로 넘어감)."""
    try:
        summary_raw = call_llm(SUMMARIZE_SYSTEM_PROMPT, build_summarize_user_prompt(query, chunks), provider=provider)
        summary_json = parse_json_response(summary_raw)
        summary = summary_json["summary"]
        citations = summary_json.get("citations", [])

        grounding_raw = call_llm(GROUNDING_SYSTEM_PROMPT, build_grounding_user_prompt(summary, chunks), provider=provider)
        grounding = parse_json_response(grounding_raw)

        judge_raw = call_llm(JUDGE_SYSTEM_PROMPT, build_judge_user_prompt(query, summary, grounding), provider=provider)
        judge = parse_json_response(judge_raw)
    except (LLMError, KeyError, ValueError, TypeError):
        return None

    grounding_passed = bool(grounding.get("passed", False))
    score = float(judge.get("score", 0))
    return AttemptResult(
        attempt_number=0,  # 호출부가 채운다
        summary=summary,
        citations=citations,
        grounding_passed=grounding_passed,
        grounding_issues=grounding.get("issues", []),
        judge_score=score,
        judge_reasoning=judge.get("reasoning", ""),
        passed_threshold=grounding_passed and score >= score_threshold,
    )


def _failed_parse_attempt() -> AttemptResult:
    return AttemptResult(
        attempt_number=0,
        summary="",
        citations=[],
        grounding_passed=False,
        grounding_issues=[],
        judge_score=0,
        judge_reasoning="응답 파싱 실패",
        passed_threshold=False,
    )


def summarize_with_verification(
    query: str,
    chunks: list[dict],
    *,
    max_attempts: int | None = None,
    score_threshold: float | None = None,
    provider: str | None = None,
) -> SummarizeResult:
    max_attempts = max_attempts or config.RAG_SUMMARIZE_MAX_ATTEMPTS
    score_threshold = score_threshold if score_threshold is not None else config.RAG_SUMMARIZE_SCORE_THRESHOLD
    provider = provider or config.RAG_SUMMARIZE_PROVIDER
    run_id = str(uuid.uuid4())

    if not chunks:
        return SummarizeResult(run_id=run_id, query=query, attempts=[])

    attempts: list[AttemptResult] = []
    for attempt_number in range(1, max_attempts + 1):
        result = _run_one_attempt(query, chunks, provider, score_threshold)
        if result is None:
            result = _failed_parse_attempt()
        result.attempt_number = attempt_number
        attempts.append(result)

        save_agent_run(
            run_id=run_id, query=query, attempt_number=attempt_number, summary=result.summary,
            citations=result.citations, grounding_passed=result.grounding_passed,
            grounding_issues=result.grounding_issues, judge_score=result.judge_score,
            judge_reasoning=result.judge_reasoning, provider=provider,
            passed_threshold=result.passed_threshold,
        )

        if result.passed_threshold:
            break

    return SummarizeResult(run_id=run_id, query=query, attempts=attempts)
