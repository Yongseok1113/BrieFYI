# rag_latest 요약+검증+판정 에이전트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rag_latest/agent_tool.py`의 `search_news()` 검색 결과를 받아 요약을 생성하고, grounding 검증과 LLM-as-a-judge 점수를 매겨, 임계치 미달이면 재생성하는 루프를 `rag_latest/summarize_agent.py`에 구현한다. 매 시도는 DB에 개별 행으로 기록한다.

**Architecture:** 스펙 문서(`docs/superpowers/specs/2026-08-18-rag-summarize-verify-judge-agent-design.md`)를 그대로 구현한다 — `llm_client.py`(Groq/Anthropic 스위처블) → `prompts.py`(요약/검증/판정 3종) → `summarize_agent.py`(오케스트레이션+재시도 루프) → `db.py`(매 시도 기록).

**Tech Stack:** Python 3.11, `groq`/`anthropic` SDK, 레포 루트 `db/db.py`(psycopg), `unittest`.

## Global Constraints

- `rag/`, `rag_experiment/`, `rag_latest/agent_tool.py`·`retriever.py`·`reranker.py`는 수정하지 않는다(이 계획은 새 모듈만 추가한다).
- 각 태스크는 별도 커밋으로 남긴다. 커밋 메시지는 한국어로 작성한다.
- 새 DB 테이블은 `db/vector_schema.sql`에 추가한다(RAG 관련 스키마가 모이는 곳).
- 빌드/개발 단계 기본 LLM provider는 **Groq**다(`RAG_SUMMARIZE_PROVIDER` 환경변수 기본값).

---

### Task 1: `config.py`(루트) 설정 추가

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Test: `tests/test_config_rag_summarize.py` (신규)

**Interfaces:**
- Produces: `config.GROQ_API_KEY: str`, `config.GROQ_MODEL: str`, `config.RAG_SUMMARIZE_PROVIDER: str`, `config.RAG_SUMMARIZE_MAX_ATTEMPTS: int`, `config.RAG_SUMMARIZE_SCORE_THRESHOLD: float`

**배경:** `GROQ_API_KEY`는 `data_pipeline/`이 이미 같은 `.env`에서 읽고 있는 값을 공유한다(별도 프리픽스 없음, `.env.example`에도 이미 있음). `GROQ_MODEL`은 새 변수 — `data_pipeline`의 `DATA_PIPELINE_GROQ_MODEL`과 이름이 겹치지 않는다(그쪽은 완전히 별도 `config.py`를 쓰므로 애초에 충돌 여지가 없다).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_config_rag_summarize.py
from config import config


def test_groq_설정_기본값():
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"


def test_rag_summarize_설정_기본값():
    assert config.RAG_SUMMARIZE_PROVIDER == "groq"
    assert config.RAG_SUMMARIZE_MAX_ATTEMPTS == 3
    assert config.RAG_SUMMARIZE_SCORE_THRESHOLD == 70.0
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m unittest tests.test_config_rag_summarize -v
```

Expected: FAIL — `AttributeError: 'Config' object has no attribute 'GROQ_MODEL'`.

- [ ] **Step 3: `config.py`에 추가** (`# Discord` 섹션과 `# 파이프라인 파라미터` 섹션 사이에 삽입)

```python
    # Groq (rag_latest/summarize_agent.py 빌드/개발 단계 기본 provider. data_pipeline과
    # 같은 GROQ_API_KEY를 공유하되 모델명은 별도 변수로 둔다 — data_pipeline은 완전히
    # 별도 config.py를 쓰므로 이름 충돌 없음)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # rag_latest/summarize_agent.py 재시도 루프 설정
    RAG_SUMMARIZE_PROVIDER = os.getenv("RAG_SUMMARIZE_PROVIDER", "groq")  # "groq" | "anthropic"
    RAG_SUMMARIZE_MAX_ATTEMPTS = int(os.getenv("RAG_SUMMARIZE_MAX_ATTEMPTS", "3"))
    RAG_SUMMARIZE_SCORE_THRESHOLD = float(os.getenv("RAG_SUMMARIZE_SCORE_THRESHOLD", "70"))
```

- [ ] **Step 4: `.env.example`에 추가** (파일 끝에)

```
# rag_latest/summarize_agent.py — 검색 결과 요약 + grounding 검증 + LLM judge 재시도 루프.
# GROQ_API_KEY는 위 data_pipeline 섹션의 값을 그대로 공유한다.
GROQ_MODEL="llama-3.3-70b-versatile"
RAG_SUMMARIZE_PROVIDER=groq
RAG_SUMMARIZE_MAX_ATTEMPTS=3
RAG_SUMMARIZE_SCORE_THRESHOLD=70
```

- [ ] **Step 5: 통과 확인**

```bash
python -m unittest tests.test_config_rag_summarize -v
```

Expected: 2 passed.

- [ ] **Step 6: 회귀 확인**

```bash
python -m unittest tests.test_main_modes tests.test_scheduler -v
```

Expected: 기존과 동일하게 통과(설정 추가만으로는 다른 동작에 영향 없음).

- [ ] **Step 7: Commit**

```bash
git add config.py .env.example tests/test_config_rag_summarize.py
git commit -m "feat(rag_latest): summarize_agent용 Groq/재시도 설정 추가"
```

---

### Task 2: `db/vector_schema.sql` — `summarize_agent_runs` 테이블

**Files:**
- Modify: `db/vector_schema.sql`
- Test: `rag_latest/tests/test_db_agent_runs.py` (신규)

**Interfaces:**
- Produces: `summarize_agent_runs` 테이블 (컬럼: `id, run_id, query, attempt_number, summary, citations, grounding_passed, grounding_issues, judge_score, judge_reasoning, provider, passed_threshold, created_at`)

- [ ] **Step 1: 현재 스키마 확인**

```bash
docker exec briefyi-db-1 psql -U briefyi -d briefyi -c "\dt" 2>&1
```

Expected: `summarize_agent_runs`가 아직 없음.

- [ ] **Step 2: `db/vector_schema.sql` 끝에 추가**

```sql
-- 검색 결과 요약 + grounding 검증 + LLM judge 재시도 루프의 시도별 기록
-- (rag_latest/summarize_agent.py). run_id로 한 번의 호출에 속한 여러 attempt를 묶는다.
CREATE TABLE IF NOT EXISTS summarize_agent_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    query TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    summary TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    grounding_passed BOOLEAN NOT NULL,
    grounding_issues JSONB NOT NULL DEFAULT '[]',
    judge_score DOUBLE PRECISION NOT NULL CHECK (judge_score BETWEEN 0 AND 100),
    judge_reasoning TEXT,
    provider TEXT NOT NULL,
    passed_threshold BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS summarize_agent_runs_run_id_idx ON summarize_agent_runs (run_id);
```

- [ ] **Step 3: 실패하는 테스트 작성**

```python
# rag_latest/tests/test_db_agent_runs.py
import unittest
import uuid

from db.db import get_conn, init_db


class SummarizeAgentRunsSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 연결 불가: {exc}")

    def test_테이블에_INSERT하고_조회할_수_있다(self):
        run_id = str(uuid.uuid4())
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO summarize_agent_runs
                   (run_id, query, attempt_number, summary, citations, grounding_passed,
                    grounding_issues, judge_score, judge_reasoning, provider, passed_threshold)
                   VALUES (%s, 'test query', 1, 'test summary', '[]', true, '[]', 85.0,
                           'ok', 'groq', true)""",
                (run_id,),
            )
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM summarize_agent_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        self.assertEqual(row["query"], "test query")
        self.assertTrue(row["passed_threshold"])
        with get_conn() as conn:
            conn.execute("DELETE FROM summarize_agent_runs WHERE run_id = %s", (run_id,))

    def test_judge_score_범위_제약이_동작한다(self):
        import psycopg

        run_id = str(uuid.uuid4())
        with self.assertRaises(psycopg.errors.CheckViolation):
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO summarize_agent_runs
                       (run_id, query, attempt_number, summary, grounding_passed,
                        judge_score, provider, passed_threshold)
                       VALUES (%s, 'q', 1, 's', true, 150.0, 'groq', true)""",
                    (run_id,),
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: 실행해 실패 확인**

```bash
docker compose build db && docker compose up -d db   # Dockerfile.db가 vector_schema.sql을 다시 COPY
python -c "from db.db import init_db; init_db()"       # 기존 볼륨 유지 시 idempotent 재적용
python -m unittest rag_latest.tests.test_db_agent_runs -v
```

Expected: `docker compose build`가 이미지를 갱신하지 않아도(볼륨 보존 시) `init_db()`가 `CREATE TABLE IF NOT EXISTS`로 새 테이블을 만들어주므로, 재빌드 없이도 이 테스트는 통과해야 한다 — 통과 확인 (2 passed).

- [ ] **Step 5: Commit**

```bash
git add db/vector_schema.sql rag_latest/tests/test_db_agent_runs.py
git commit -m "feat(rag_latest): summarize_agent_runs 테이블 추가"
```

---

### Task 3: `rag_latest/llm_client.py`

**Files:**
- Create: `rag_latest/llm_client.py`
- Test: `rag_latest/tests/test_llm_client.py`

**Interfaces:**
- Produces: `call_llm(system: str, user: str, *, max_tokens: int = 1500, provider: str | None = None) -> str`, `parse_json_response(text: str) -> dict | list`, `class LLMError(RuntimeError)`

- [ ] **Step 1: 실패하는 테스트 작성** (`parse_json_response`만 — 순수 함수, `call_llm`은 실제 API 필요하므로 여기서는 provider 라우팅만 mock으로 검증)

```python
# rag_latest/tests/test_llm_client.py
import unittest
from unittest.mock import patch

from rag_latest.llm_client import LLMError, call_llm, parse_json_response


class ParseJsonResponseTest(unittest.TestCase):
    def test_코드블록으로_감싼_JSON을_파싱한다(self):
        text = '설명입니다\n```json\n{"score": 85}\n```\n'
        self.assertEqual(parse_json_response(text), {"score": 85})

    def test_코드블록_없이_바로_JSON이면_그대로_파싱한다(self):
        self.assertEqual(parse_json_response('{"passed": true}'), {"passed": True})

    def test_JSON_뒤에_부연설명이_붙어도_첫_JSON만_파싱한다(self):
        text = '{"summary": "x"} 이상입니다.'
        self.assertEqual(parse_json_response(text), {"summary": "x"})


class CallLlmProviderRoutingTest(unittest.TestCase):
    def test_지원하지_않는_provider면_LLMError(self):
        with self.assertRaises(LLMError):
            call_llm("sys", "user", provider="does-not-exist")

    def test_provider_생략하면_config_기본값을_쓴다(self):
        with patch("rag_latest.llm_client.config") as mock_config, \
             patch("rag_latest.llm_client._call_groq", return_value="ok") as mock_groq:
            mock_config.RAG_SUMMARIZE_PROVIDER = "groq"
            result = call_llm("sys", "user")
        mock_groq.assert_called_once()
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m unittest rag_latest.tests.test_llm_client -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag_latest.llm_client'`.

- [ ] **Step 3: `rag_latest/llm_client.py` 작성**

```python
"""요약/검증/판정용 LLM 호출. data_pipeline/src/data_pipeline/llm_client.py와 동일한
패턴(공용 call_llm, provider별 lazy 클라이언트)을 최소 기능만 가져와 새로 만든다 —
data_pipeline/을 import하지 않는다(별도 배포 이미지). groq/anthropic 두 provider만
지원한다(hf는 이 용도에 불필요).
"""
from __future__ import annotations

import json
import re

from config import config

_groq_client = None
_anthropic_client = None


class LLMError(RuntimeError):
    pass


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        if not config.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _call_groq(system: str, user: str, max_tokens: int) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


_PROVIDERS = {"groq": _call_groq, "anthropic": _call_anthropic}


def call_llm(system: str, user: str, *, max_tokens: int = 1500, provider: str | None = None) -> str:
    provider = provider or config.RAG_SUMMARIZE_PROVIDER
    call_fn = _PROVIDERS.get(provider)
    if call_fn is None:
        raise LLMError(f"지원하지 않는 provider: {provider} (groq/anthropic 중 하나)")
    try:
        return call_fn(system, user, max_tokens)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider별 예외 타입이 달라 폭넓게 잡는다
        raise LLMError(f"LLM 호출 실패 ({provider}): {exc}") from exc


def parse_json_response(text: str) -> dict | list:
    """```json 코드블록 우선, 없으면 첫 '{'/'['부터 raw_decode — data_pipeline/llm_client.py와
    동일한 관용구(모델이 JSON 앞뒤에 설명을 덧붙이는 경우까지 안전하게 처리)."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    payload = (match.group(1) if match else text).strip()

    starts = [i for i in (payload.find("{"), payload.find("[")) if i != -1]
    start = min(starts) if starts else 0

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(payload[start:])
    return obj
```

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest rag_latest.tests.test_llm_client -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rag_latest/llm_client.py rag_latest/tests/test_llm_client.py
git commit -m "feat(rag_latest): Groq/Anthropic 스위처블 LLM 클라이언트 추가"
```

---

### Task 4: `rag_latest/prompts.py`

**Files:**
- Create: `rag_latest/prompts.py`
- Test: `rag_latest/tests/test_prompts.py`

**Interfaces:**
- Produces: `SUMMARIZE_SYSTEM_PROMPT: str`, `GROUNDING_SYSTEM_PROMPT: str`, `JUDGE_SYSTEM_PROMPT: str`, `build_summarize_user_prompt(query: str, chunks: list[dict]) -> str`, `build_grounding_user_prompt(summary: str, chunks: list[dict]) -> str`, `build_judge_user_prompt(query: str, summary: str, grounding_result: dict) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# rag_latest/tests/test_prompts.py
import unittest

from rag_latest.prompts import (
    build_grounding_user_prompt,
    build_judge_user_prompt,
    build_summarize_user_prompt,
)


class BuildPromptsTest(unittest.TestCase):
    def test_summarize_prompt는_질의와_chunk를_포함한다(self):
        chunks = [{"article_id": 1, "title": "제목1", "url": "https://x/1", "text": "본문1", "category": "기술"}]
        prompt = build_summarize_user_prompt("Claude 관련 뉴스", chunks)
        self.assertIn("Claude 관련 뉴스", prompt)
        self.assertIn("제목1", prompt)
        self.assertIn("본문1", prompt)
        self.assertIn("1", prompt)  # article_id 인용 가능하도록 노출

    def test_grounding_prompt는_요약과_chunk를_포함한다(self):
        chunks = [{"article_id": 1, "title": "제목1", "text": "본문1"}]
        prompt = build_grounding_user_prompt("이것은 요약입니다", chunks)
        self.assertIn("이것은 요약입니다", prompt)
        self.assertIn("본문1", prompt)

    def test_judge_prompt는_질의_요약_grounding결과를_포함한다(self):
        grounding_result = {"passed": True, "issues": []}
        prompt = build_judge_user_prompt("질의", "요약문", grounding_result)
        self.assertIn("질의", prompt)
        self.assertIn("요약문", prompt)
        self.assertIn("true", prompt.lower())

    def test_빈_chunk_목록도_에러_없이_처리한다(self):
        prompt = build_summarize_user_prompt("질의", [])
        self.assertIn("질의", prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m unittest rag_latest.tests.test_prompts -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `rag_latest/prompts.py` 작성**

```python
"""요약/검증(grounding)/판정(judge) 프롬프트 3종.

tools/summarize.py, tools/insight.py와 같은 스타일 — 시스템 프롬프트는 모듈 상수,
JSON-only 응답을 강제한다. summarize_agent.py가 이 모듈의 함수로 user 프롬프트를
조합만 하도록 로직 없이 문자열 구성만 담당한다.
"""
from __future__ import annotations

import json

SUMMARIZE_SYSTEM_PROMPT = """당신은 뉴스 검색 결과를 종합해 질의에 답하는 리서치 어시스턴트다.
주어진 chunk 목록만 근거로 삼아 질의에 답하는 요약문을 작성한다. chunk에 없는 정보는
절대 추측하지 않는다. 요약문의 각 문장은 근거가 된 chunk의 article_id를 인용해야 한다.
반드시 아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
{
  "summary": "질의에 답하는 요약문",
  "citations": [{"sentence": "요약문 속 문장", "article_id": 123}]
}
```
"""

GROUNDING_SYSTEM_PROMPT = """당신은 사실 검증 담당자다. 주어진 요약문의 각 문장이 원본
chunk에 실제로 근거하는지 대조한다. chunk에 없는 주장, 과잉 해석, 사실과 다른 서술을
모두 찾아낸다.
반드시 아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
{
  "passed": true,
  "issues": [{"sentence": "문제가 된 문장", "reason": "근거 없음 | 과잉해석 | 사실과 다름"}]
}
```

issues가 비어 있으면 passed는 true, 하나라도 있으면 passed는 false여야 한다.
"""

JUDGE_SYSTEM_PROMPT = """당신은 요약 품질을 평가하는 심사자(LLM-as-a-judge)다. 질의, 요약문,
grounding 검증 결과를 보고 0~100점으로 종합 평가한다. 평가 기준은 4개, 각 0~25점:
근거충실성(grounding 결과를 반영, 근거 없는 문장이 있으면 감점), 질의 관련성(질의에
실제로 답하는가), 완전성(핵심 정보 누락 없음), 간결성(불필요하게 장황하지 않음).
반드시 아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
{
  "score": 82,
  "reasoning": "종합 평가 이유",
  "breakdown": {"grounding": 22, "relevance": 20, "completeness": 20, "conciseness": 20}
}
```
"""


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(검색 결과 없음)"
    return "\n\n".join(
        f"[article_id={c.get('article_id')}] {c.get('title', '')}\n{c.get('text', '')}"
        for c in chunks
    )


def build_summarize_user_prompt(query: str, chunks: list[dict]) -> str:
    return f"질의: {query}\n\n검색된 chunk 목록:\n{_format_chunks(chunks)}"


def build_grounding_user_prompt(summary: str, chunks: list[dict]) -> str:
    return f"요약문:\n{summary}\n\n원본 chunk 목록:\n{_format_chunks(chunks)}"


def build_judge_user_prompt(query: str, summary: str, grounding_result: dict) -> str:
    return (
        f"질의: {query}\n\n요약문:\n{summary}\n\n"
        f"grounding 검증 결과:\n{json.dumps(grounding_result, ensure_ascii=False)}"
    )
```

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest rag_latest.tests.test_prompts -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add rag_latest/prompts.py rag_latest/tests/test_prompts.py
git commit -m "feat(rag_latest): 요약/grounding검증/judge 프롬프트 3종 추가"
```

---

### Task 5: `rag_latest/db.py` — `save_agent_run()` 추가

**Files:**
- Modify: `rag_latest/db.py`
- Test: `rag_latest/tests/test_db.py` (기존 파일에 테스트 추가)

**Interfaces:**
- Consumes: Task 2의 `summarize_agent_runs` 테이블
- Produces: `save_agent_run(run_id: str, query: str, attempt_number: int, summary: str, citations: list[dict], grounding_passed: bool, grounding_issues: list[dict], judge_score: float, judge_reasoning: str, provider: str, passed_threshold: bool) -> None`

- [ ] **Step 1: 실패하는 테스트 작성** (기존 `rag_latest/tests/test_db.py` 맨 끝에 추가)

```python
class SaveAgentRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 연결 불가: {exc}")

    def test_한_행이_저장되고_조회된다(self):
        import uuid

        from rag_latest.db import save_agent_run

        run_id = str(uuid.uuid4())
        save_agent_run(
            run_id=run_id, query="테스트 질의", attempt_number=1, summary="테스트 요약",
            citations=[{"sentence": "s", "article_id": 1}], grounding_passed=True,
            grounding_issues=[], judge_score=88.5, judge_reasoning="근거 충분",
            provider="groq", passed_threshold=True,
        )
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM summarize_agent_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        self.assertEqual(row["query"], "테스트 질의")
        self.assertEqual(row["judge_score"], 88.5)
        with get_conn() as conn:
            conn.execute("DELETE FROM summarize_agent_runs WHERE run_id = %s", (run_id,))
```

(파일 상단에 이미 `import unittest`, `from db.db import get_conn, init_db`가 있는지 확인 — 없으면 추가)

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m unittest rag_latest.tests.test_db -v
```

Expected: FAIL — `ImportError: cannot import name 'save_agent_run'`.

- [ ] **Step 3: `rag_latest/db.py` 끝에 추가**

```python
def save_agent_run(
    run_id: str,
    query: str,
    attempt_number: int,
    summary: str,
    citations: list[dict],
    grounding_passed: bool,
    grounding_issues: list[dict],
    judge_score: float,
    judge_reasoning: str,
    provider: str,
    passed_threshold: bool,
) -> None:
    """summarize_agent.py의 시도 1건을 summarize_agent_runs에 기록한다."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO summarize_agent_runs
                   (run_id, query, attempt_number, summary, citations, grounding_passed,
                    grounding_issues, judge_score, judge_reasoning, provider, passed_threshold)
               VALUES (%(run_id)s, %(query)s, %(attempt_number)s, %(summary)s, %(citations)s,
                       %(grounding_passed)s, %(grounding_issues)s, %(judge_score)s,
                       %(judge_reasoning)s, %(provider)s, %(passed_threshold)s)""",
            {
                "run_id": run_id,
                "query": query,
                "attempt_number": attempt_number,
                "summary": summary,
                "citations": Jsonb(citations),
                "grounding_passed": grounding_passed,
                "grounding_issues": Jsonb(grounding_issues),
                "judge_score": judge_score,
                "judge_reasoning": judge_reasoning,
                "provider": provider,
                "passed_threshold": passed_threshold,
            },
        )
```

파일 상단 import에 `from psycopg.types.json import Jsonb` 추가(아직 없으면).

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest rag_latest.tests.test_db -v
```

Expected: 기존 테스트 + 신규 1개 전부 통과.

- [ ] **Step 5: Commit**

```bash
git add rag_latest/db.py rag_latest/tests/test_db.py
git commit -m "feat(rag_latest): save_agent_run() 추가 — 요약 에이전트 시도별 기록"
```

---

### Task 6: `rag_latest/summarize_agent.py`

**Files:**
- Create: `rag_latest/summarize_agent.py`
- Test: `rag_latest/tests/test_summarize_agent.py`

**Interfaces:**
- Consumes: `llm_client.call_llm`, `llm_client.parse_json_response`, `prompts.*`, `db.save_agent_run`
- Produces: `AttemptResult`(dataclass), `SummarizeResult`(dataclass, `.passed`/`.final` 프로퍼티), `summarize_with_verification(query: str, chunks: list[dict], *, max_attempts=None, score_threshold=None, provider=None) -> SummarizeResult`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# rag_latest/tests/test_summarize_agent.py
import unittest
from unittest.mock import patch

from rag_latest.summarize_agent import summarize_with_verification

CHUNKS = [{"article_id": 1, "title": "제목", "url": "https://x/1", "text": "본문"}]


def _llm_responses(*json_strings):
    """call_llm이 호출 순서대로 반환할 문자열 시퀀스. 한 시도당 summarize->grounding->judge
    3번 호출되므로, 시도 N개면 길이 3N짜리 시퀀스를 넘긴다."""
    return list(json_strings)


class SummarizeWithVerificationTest(unittest.TestCase):
    def test_1회차에_바로_통과하면_1개_시도만_기록된다(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 85, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses) as mock_call, \
             patch("rag_latest.summarize_agent.save_agent_run") as mock_save:
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(mock_call.call_count, 3)
        mock_save.assert_called_once()

    def test_1회차_미달_2회차_통과(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 50, "reasoning": "부족", "breakdown": {}}',
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 90, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run") as mock_save:
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.final.judge_score, 90)
        self.assertEqual(mock_save.call_count, 2)

    def test_max_attempts_소진하면_실패_플래그로_반환한다(self):
        one_attempt = [
            '{"summary": "요약", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 40, "reasoning": "부족", "breakdown": {}}',
        ]
        responses = one_attempt * 2  # max_attempts=2
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=2, score_threshold=70)

        self.assertFalse(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.final.passed_threshold)

    def test_grounding_실패하면_점수가_높아도_재시도한다(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": false, "issues": [{"sentence": "x", "reason": "근거 없음"}]}',
            '{"score": 99, "reasoning": "점수는 높음", "breakdown": {}}',
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 80, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run"):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].passed_threshold)  # grounding 실패 + 점수 99여도 미달 처리

    def test_빈_chunk는_LLM_호출_없이_즉시_실패_반환한다(self):
        with patch("rag_latest.summarize_agent.call_llm") as mock_call:
            result = summarize_with_verification("질의", [], max_attempts=3, score_threshold=70)

        mock_call.assert_not_called()
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, [])

    def test_JSON_파싱_실패해도_루프가_죽지_않고_다음_시도로_넘어간다(self):
        responses = [
            "이건 JSON이 아님",  # summarize 파싱 실패
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 90, "reasoning": "좋음", "breakdown": {}}',
        ]
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run"):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].passed_threshold)
        self.assertEqual(result.attempts[0].judge_score, 0)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m unittest rag_latest.tests.test_summarize_agent -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `rag_latest/summarize_agent.py` 작성**

```python
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
```

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest rag_latest.tests.test_summarize_agent -v
```

Expected: 6 passed.

- [ ] **Step 5: 전체 rag_latest 테스트 회귀 확인**

```bash
python -m unittest discover -s rag_latest/tests -t . -p "test_*.py"
```

Expected: 기존 테스트 전부 + 신규 테스트 전부 통과.

- [ ] **Step 6: Commit**

```bash
git add rag_latest/summarize_agent.py rag_latest/tests/test_summarize_agent.py
git commit -m "feat(rag_latest): 요약+grounding검증+LLM judge 재시도 루프 오케스트레이션 추가"
```

---

## 실행 후 전체 회귀 확인 (모든 태스크 완료 후)

```bash
python -m unittest discover -t .                                     # 루트 + rag/tests
python -m unittest discover -s rag_latest/tests -t . -p "test_*.py"   # rag_latest (신규 포함)
```

`GROQ_API_KEY`가 `.env`에 설정돼 있으면(이미 있음, 이번 통합 수정에서 `Groq`→`groq` 오타도 고쳤음) 실제 Groq 호출 없이도(모두 mock) 위 명령으로 전체 회귀를 확인할 수 있다. 실제 Groq 호출로 end-to-end 확인하려면:

```bash
python -c "
from rag_latest.agent_tool import search_news
from rag_latest.summarize_agent import summarize_with_verification

chunks = search_news('Claude', top_k=5)
result = summarize_with_verification('Claude 관련 최근 소식은?', chunks)
print('passed:', result.passed, '시도 횟수:', len(result.attempts))
print(result.final.summary)
"
```
