# BrieFYI 통합 수정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/project-status-2026-08.md`에서 확인된 10개 통합 이슈(문서 링크 깨짐, 의존성 누락, 버그 2건, 고아 모듈 정리, 인프라 이미지 불일치 2건, `rag-latest/` 통합 신규 구축)를 한 번에 몰아치지 않고 태스크 단위로 순차 적용해, 각 태스크가 끝날 때마다 기존 테스트로 회귀 여부를 확인할 수 있게 한다.

**Architecture:** 태스크는 위험도가 낮고 서로 독립적인 것부터 순서대로 배치했다(문서/의존성 정리 → 격리된 버그 수정 → 활성 발송 경로 수정 → 인프라 이미지 → 마지막으로 가장 큰 신규 작업인 `rag_latest/` 통합). 앞 태스크의 결과가 뒤 태스크의 전제조건이 되는 경우는 없으므로, 필요하면 순서를 건너뛰거나 일부만 적용해도 안전하다. 단, 8번(로컬 DB 재빌드)과 9번(CI 이미지 교체)은 같은 근본 원인(pgvector 미포함 이미지)이라 함께 검토하는 걸 권장한다.

**Tech Stack:** Python 3.11, `unittest`(루트 표준) / `pytest`(`data_pipeline/`, `finetune/`), PostgreSQL 16 + pgvector, Docker Compose, GitHub Actions.

## Global Constraints

- `rag/`와 `rag_experiment/`는 **절대 수정하지 않는다** — 사용자가 추후 별도로 삭제할 예정이며 이번 계획에서는 참고용 읽기 전용 소스다.
- 기존 README/문서에 명시된 CLI·테스트 명령(`python -m unittest discover -t .` 등)을 깨지 않는다.
- 새로 추가하는 무거운 로컬 모델 의존성(`FlagEmbedding` 등)은 공유 `requirements.txt`에 넣지 않고, `rag/requirements-event.txt`처럼 별도 optional 파일로 분리한다(기존 관례).
- 각 태스크는 별도 커밋으로 남기고, 커밋 전 반드시 해당 태스크의 검증 명령을 실행해 결과를 확인한다.
- 주석/커밋 메시지는 기존 코드베이스와 동일하게 한국어로 작성한다.
- `python-dotenv`가 로드하는 실제 `.env`(로컬 값)는 건드리지 않는다. `.env.example`만 갱신한다.

---

## Task 1: 깨진 문서 링크 정정 (docs/LORA-F~1.MD, docs/AGENT-~1.MD)

**Files:**
- Rename: `docs/LORA-F~1.MD` → `docs/lora-finetune-summarization-design.md`
- Rename: `docs/AGENT-~1.MD` → `docs/agent-management-structure.md`
- 참조 확인만 (수정 없음): `finetune/README.md:5`, `finetune/docs/ARCHITECTURE.md:3,44`, `agents/base.py:6`

**배경:** 두 파일은 Windows 8.3 단축 파일명 형태(`LORA-F~1.MD`, `AGENT-~1.MD`)로 잘못 커밋돼 있다. `finetune/README.md`·`finetune/docs/ARCHITECTURE.md`·`agents/base.py`의 주석은 이미 올바른 전체 파일명(`docs/lora-finetune-summarization-design.md`, `agent-management-structure.md`)을 참조하고 있으므로, 파일명만 그 이름에 맞춰 되돌리면 참조가 저절로 맞아떨어진다.

- [ ] **Step 1: 현재 깨진 참조 확인**

```bash
grep -rn "lora-finetune-summarization-design.md\|agent-management-structure.md" --include='*.md' --include='*.py' .
ls docs/ | grep '~'
```

Expected: 두 문서명을 참조하는 곳은 있지만 `docs/` 안에 그 이름의 실제 파일은 없고, 대신 `LORA-F~1.MD`/`AGENT-~1.MD`가 존재.

- [ ] **Step 2: git mv로 파일명 정정**

```bash
git mv "docs/LORA-F~1.MD" "docs/lora-finetune-summarization-design.md"
git mv "docs/AGENT-~1.MD" "docs/agent-management-structure.md"
```

- [ ] **Step 3: 참조가 모두 해소됐는지 재확인**

```bash
grep -rn "lora-finetune-summarization-design.md\|agent-management-structure.md" --include='*.md' --include='*.py' .
```

Expected: 모든 참조 경로에 실제로 파일이 존재.

- [ ] **Step 4: Commit**

```bash
git add docs/lora-finetune-summarization-design.md docs/agent-management-structure.md
git commit -m "docs: 8.3 단축 파일명으로 잘못 커밋된 설계 문서 2개 이름 정정"
```

---

## Task 2: `pipeline.db` 잔재 삭제

**Files:**
- Delete: `pipeline.db` (0바이트, PostgreSQL 전환 이전 SQLite 잔재)

**배경:** 현재 어떤 코드에서도 참조하지 않는 빈 SQLite 파일이다. `db/db.py`는 PostgreSQL(`psycopg`)만 쓴다.

- [ ] **Step 1: 미참조 확인**

```bash
grep -rln "pipeline\.db" --include='*.py' --include='*.yml' --include='*.yaml' . | grep -v .venv
```

Expected: 결과 없음.

- [ ] **Step 2: 삭제 및 재생성 방지**

```bash
git rm pipeline.db
```

`.gitignore`에 이미 `*.pyc`/`__pycache__` 등은 있지만 `*.db` 패턴은 없다. 같은 실수가 반복되지 않도록 추가한다.

```
# .gitignore 끝에 추가
*.db
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: PostgreSQL 전환 이전 SQLite 잔재(pipeline.db) 삭제, *.db gitignore 추가"
```

---

## Task 3: 루트 `pyproject.toml` 의존성 보강

**Files:**
- Modify: `pyproject.toml`

**배경:** `db/db.py`가 실제로 쓰는 `psycopg`·`pgvector`가 루트 `pyproject.toml`의 `dependencies`에 빠져 있다. Docker 이미지는 `requirements.txt`를 쓰므로 영향 없지만(`Dockerfile:6-7`), `uv sync`로 로컬 환경을 새로 만들면 DB 관련 코드가 즉시 `ModuleNotFoundError`가 난다. `requirements.txt`가 실제 배포 기준이므로 그쪽 버전 하한을 그대로 따른다.

- [ ] **Step 1: 현재 불일치 확인**

```bash
diff <(grep -oE '^[a-zA-Z_-]+' requirements.txt | sort) \
     <(python -c "import tomllib; print('\n'.join(sorted(d.split('[')[0].split('>=')[0].strip() for d in tomllib.load(open('pyproject.toml','rb'))['project']['dependencies'])))")
```

Expected: `requirements.txt`에는 있는데 `pyproject.toml`에는 없는 항목으로 `psycopg[binary]`, `pgvector`가 나온다.

- [ ] **Step 2: `pyproject.toml`에 누락 의존성 추가**

`pyproject.toml`의 `dependencies` 배열을 다음으로 교체한다(기존 6개 + `requirements.txt`에서 가져온 2개, 버전 하한은 `requirements.txt` 기준):

```toml
dependencies = [
    "anthropic>=0.121.0",
    "jinja2>=3.1.6",
    "langgraph>=1.2.11",
    "pgvector>=0.3.0",
    "psycopg[binary]>=3.1.0",
    "python-dotenv>=1.2.2",
    "requests>=2.34.2",
    "typing-extensions>=4.16.0",
]
```

- [ ] **Step 3: `uv sync`로 재검증**

```bash
uv sync
uv pip list --python .venv/bin/python | grep -iE "psycopg|pgvector"
```

Expected: 둘 다 설치됨으로 표시.

- [ ] **Step 4: 기존 테스트로 회귀 없는지 확인**

```bash
python -m unittest tests.test_main_modes tests.test_scheduler
```

Expected: 기존과 동일하게 통과(이 두 테스트는 DB 접속 불필요, mock 기반).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: requirements.txt에는 있고 pyproject.toml에는 없던 psycopg/pgvector 추가"
```

---

## Task 4: `tools/discord_send.py` — `config.DISCORD_WEBHOOK_URL` 미정의 버그 수정

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Test: `tests/test_discord_send.py` (신규)

**Interfaces:**
- Consumes: `tools/discord_send.py:79`의 기존 `config.DISCORD_WEBHOOK_URL` 참조(변경 없음, 속성만 채워 넣음)
- Produces: `config.DISCORD_WEBHOOK_URL: str` — 향후 `agents/registry.py`에 discord 채널을 등록할 때 그대로 재사용

**배경:** `tools/discord_send.py:79`가 `config.DISCORD_WEBHOOK_URL`을 참조하지만 `config.py`의 `Config` 클래스에 해당 속성이 없어, 실제로 호출되면 의도한 `RuntimeError`(".env 확인") 대신 `AttributeError`가 난다. 지금은 `agents/distributor.py`의 `channels`에 `"discord"`가 등록돼 있지 않아 죽은 코드지만, 나중에 Discord 채널을 켤 때 바로 쓸 수 있도록 지금 고쳐 둔다.

- [ ] **Step 1: 버그를 드러내는 실패 테스트 작성**

```python
# tests/test_discord_send.py
import unittest
from unittest.mock import patch

from tools.discord_send import send_discord


class DiscordWebhookConfigTest(unittest.TestCase):
    def test_웹훅_URL이_없으면_RuntimeError를_던진다(self):
        with patch("tools.discord_send.config") as mock_config:
            mock_config.DISCORD_WEBHOOK_URL = ""
            with self.assertRaises(RuntimeError) as ctx:
                send_discord("제목", summaries=[], insight={})
        self.assertIn("DISCORD_WEBHOOK_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인 (지금은 AttributeError로 실패해야 정상)**

```bash
python -m unittest tests.test_discord_send -v
```

Expected: `AttributeError: <MagicMock ...> DISCORD_WEBHOOK_URL` 계열 에러로 FAIL — `config.py`에 속성이 없어서가 아니라, mock의 속성 설정은 되지만 실제 `config` singleton에는 `DISCORD_WEBHOOK_URL` 자체가 없다는 걸 별도로 확인:

```bash
python -c "from config import config; print(getattr(config, 'DISCORD_WEBHOOK_URL', 'NOT_DEFINED'))"
```

Expected: `NOT_DEFINED` 출력.

- [ ] **Step 3: `config.py`에 속성 추가**

`config.py`의 Email(Resend) 섹션 근처(`RESEND_API_KEY` 등 아래)에 추가:

```python
    # Discord (백로그 #9, tools/discord_send.py)
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
```

- [ ] **Step 4: `.env.example`에 항목 추가**

`.env.example`의 이메일 섹션 아래에 추가:

```
# Discord 발송 (백로그 #9, tools/discord_send.py). 채널을 켜려면
# agents/registry.py의 DistributorAgent channels/tools에도 등록해야 한다.
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx/yyy"
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -c "from config import config; print(getattr(config, 'DISCORD_WEBHOOK_URL', 'NOT_DEFINED'))"
python -m unittest tests.test_discord_send -v
```

Expected: 첫 명령은 빈 문자열 `""` 출력(NOT_DEFINED 아님), 두 번째는 PASS.

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example tests/test_discord_send.py
git commit -m "fix: config.DISCORD_WEBHOOK_URL 누락으로 discord_send 호출 시 AttributeError 나던 버그 수정"
```

---

## Task 5: `finetune` YAML `learning_rate` 문자열 파싱 버그 수정

**Files:**
- Modify: `finetune/src/summarize_ft/config.py:66-84` (`_dict_to_config`)

**배경:** `finetune/configs/*.yaml` 전 파일이 `learning_rate: 1e-4` / `2e-4`처럼 소수점 없는 지수 표기를 쓰는데, PyYAML의 기본 리졸버는 이 형태를 float가 아니라 **문자열**로 파싱한다(잘 알려진 PyYAML 동작 — float 정규식이 `.`을 요구함). `_dict_to_config()`(`config.py:66-84`)는 파싱된 dict 값을 타입 캐스팅 없이 그대로 `TrainConfig(**nested_dict)`에 넘기므로, `learning_rate`가 문자열 `"1e-4"`로 들어간 채로 `TrainConfig`가 만들어지고, `validate_config()`(`config.py:160`)의 `cfg.train.learning_rate <= 0` 비교에서 `TypeError: '<=' not supported between instances of 'str' and 'int'`가 난다. 이미 존재하는 `finetune/tests/test_config.py::test_load_full_config`가 정확히 이 버그를 재현하고 있으므로, 새 테스트는 필요 없고 기존 테스트를 그린으로 만드는 게 목표다.

- [ ] **Step 1: 기존 실패 재현**

```bash
cd /home/ysoh1113/workspace/projects/BrieFYI
.venv/bin/python -m pytest finetune/tests/test_config.py::test_load_full_config -v
```

Expected: FAIL — `TypeError: '<=' not supported between instances of 'str' and 'int'` at `config.py:160`.

- [ ] **Step 2: `_dict_to_config`에 `learning_rate` 캐스팅 추가**

`finetune/src/summarize_ft/config.py`의 `_dict_to_config` 함수를 다음으로 교체:

```python
def _dict_to_config(d: dict[str, Any]) -> Config:
    d = dict(d)  # shallow copy, 원본 보존
    kwargs: dict[str, Any] = {}
    for key, nested_cls in _NESTED_TYPES.items():
        nested_dict = d.pop(key, {}) or {}
        if not isinstance(nested_dict, dict):
            raise ConfigError(f"'{key}' 섹션은 dict여야 함, got {type(nested_dict)}")
        valid_keys = {f.name for f in fields(nested_cls)}
        unknown = set(nested_dict) - valid_keys
        if unknown:
            raise ConfigError(f"'{key}' 섹션에 알 수 없는 필드: {sorted(unknown)}")
        if "learning_rate" in nested_dict:
            # PyYAML은 소수점 없는 지수 표기(예: 1e-4)를 float가 아니라 문자열로 파싱한다.
            # 모든 finetune/configs/*.yaml이 이 표기를 쓰므로 명시적으로 캐스팅한다.
            nested_dict["learning_rate"] = float(nested_dict["learning_rate"])
        kwargs[key] = nested_cls(**nested_dict)

    valid_top_keys = {f.name for f in fields(Config)}
    unknown_top = set(d) - valid_top_keys
    if unknown_top:
        raise ConfigError(f"알 수 없는 최상위 필드: {sorted(unknown_top)}")
    kwargs.update(d)
    return Config(**kwargs)
```

- [ ] **Step 3: 통과 확인**

```bash
.venv/bin/python -m pytest finetune/tests/test_config.py -v
```

Expected: 전부 PASS, `test_load_full_config`가 `cfg.train.learning_rate == 1e-4` 통과.

- [ ] **Step 4: 실제 config 파일들도 전부 로드되는지 회귀 확인**

```bash
.venv/bin/python -c "
from summarize_ft.config import load_config
import glob
for path in glob.glob('finetune/configs/*.yaml'):
    try:
        cfg = load_config(path)
        print(path, 'OK', 'lr=', cfg.train.learning_rate, type(cfg.train.learning_rate))
    except Exception as exc:
        print(path, 'FAIL', exc)
"
```

Expected: `smoke.yaml`을 포함한 모든 config가 `OK`, `learning_rate`가 `float` 타입으로 출력.

- [ ] **Step 5: finetune 전체 테스트 회귀 확인**

```bash
.venv/bin/python -m pytest finetune/tests -q
```

Expected: `57 passed` (기존 `1 failed, 56 passed`에서 실패 0건으로).

- [ ] **Step 6: Commit**

```bash
git add finetune/src/summarize_ft/config.py
git commit -m "fix: PyYAML이 1e-4 표기를 문자열로 파싱해 모든 finetune config 검증이 실패하던 버그 수정"
```

---

## Task 6: `data_pipeline` 테스트의 `.env` 의존성 격리

**Files:**
- Modify: `data_pipeline/tests/test_config.py`

**배경:** `test_llm_provider_defaults_to_groq`가 "기본값은 groq"를 검증하려 하지만, `data_pipeline/src/data_pipeline/config.py`가 모듈 임포트 시점에 실제 루트 `.env`를 읽어 싱글턴을 만든다. 로컬 `.env`에 `DATA_PIPELINE_LLM_PROVIDER=hf`가 설정돼 있으면(이 환경이 그렇다) 테스트가 항상 깨진다. 코드 버그가 아니라 테스트가 환경으로부터 격리돼 있지 않은 게 원인이므로, 해당 환경변수를 지운 뒤 모듈을 리로드해서 "정말 설정이 없을 때의 기본값"을 검증하도록 고친다.

- [ ] **Step 1: 현재 실패 재현**

```bash
cd /home/ysoh1113/workspace/projects/BrieFYI
.venv/bin/python -m pytest data_pipeline/tests/test_config.py -v
```

Expected: `test_llm_provider_defaults_to_groq` FAIL (`assert 'hf' == 'groq'`), 나머지 통과.

- [ ] **Step 2: 테스트를 환경변수로부터 격리**

`data_pipeline/tests/test_config.py` 전체를 다음으로 교체:

```python
import importlib

from data_pipeline.config import config


def test_defaults_have_expected_types():
    assert isinstance(config.RATE_LIMIT_MAX_REQUESTS, int)
    assert isinstance(config.RATE_LIMIT_WINDOW_SECONDS, float)
    assert isinstance(config.SYNONYM_CLUSTER_THRESHOLD, float)
    assert isinstance(config.SYNONYM_FUZZY_THRESHOLD, float)
    assert isinstance(config.BATCH_LIMIT, int)


def test_llm_provider_defaults_to_groq(monkeypatch):
    """DATA_PIPELINE_LLM_PROVIDER가 설정 안 됐을 때만 검증한다.

    config 모듈은 import 시점에 실제 루트 .env를 읽는 싱글턴이라, 로컬 .env에
    DATA_PIPELINE_LLM_PROVIDER가 이미 설정돼 있으면 단순히 config.LLM_PROVIDER를
    확인하는 것만으로는 "기본값"을 검증할 수 없다. 환경변수를 지우고 모듈을
    리로드해 기본값 경로를 강제로 태운 뒤, 다른 테스트에 영향이 없도록 원래
    상태로 복원한다.
    """
    from data_pipeline import config as config_module

    monkeypatch.delenv("DATA_PIPELINE_LLM_PROVIDER", raising=False)
    try:
        importlib.reload(config_module)
        assert config_module.config.LLM_PROVIDER == "groq"
    finally:
        importlib.reload(config_module)


def test_database_url_is_constructed_when_not_set_explicitly():
    assert config.DATABASE_URL.startswith("postgresql://")
```

- [ ] **Step 3: 통과 확인**

```bash
.venv/bin/python -m pytest data_pipeline/tests/test_config.py -v
```

Expected: 3개 전부 PASS.

- [ ] **Step 4: data_pipeline 전체 테스트 회귀 확인**

```bash
.venv/bin/python -m pytest data_pipeline/tests -q
```

Expected: `27 passed` (기존 `1 failed, 26 passed`에서 실패 0건으로).

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/tests/test_config.py
git commit -m "test: data_pipeline 기본값 테스트가 로컬 .env 값에 의해 깨지지 않도록 환경변수 격리"
```

---

## Task 7: `tools/deployment.py`를 `agents/distributor.py`로 흡수 (다중 수신자 지원)

**Files:**
- Modify: `config.py` (다중 수신자 파싱 추가)
- Modify: `.env.example`
- Modify: `tools/email_send.py` (`to` 파라미터 추가)
- Modify: `agents/distributor.py` (다중 수신자 루프 + 개별 로깅)
- Delete: `tools/deployment.py`
- Delete: `tests/test_deployment.py`
- Test: `tests/test_distributor.py` (신규)

**Interfaces:**
- Consumes: `config.EMAIL_TO`(기존, 변경 없음), `db.db.log_send(digest_id, channel, recipient, status, error=None)`(기존)
- Produces: `config.EMAIL_RECIPIENTS: list[str]`, `send_email(subject, html, to=None) -> dict`(신규 `to` 파라미터), `DistributorAgent.run(state) -> {"send_result": {"recipients": [...], "success_count": int, "total_count": int}}`(반환 형태 변경)

**배경(2026-08-18 대화에서 합의):** 원래 `tools/deployment.py`는 `target_adress.json`(로컬 전용, `.gitignore`에 등록돼 있으나 이 저장소엔 없음)에서 다중 수신자를 읽어 개별 발송·개별 로깅하는 기능을 갖고 있었다. 이 기능 자체는 버릴 게 아니라는 게 사용자 판단이라, **로직은 `agents/distributor.py`로 옮기고 `tools/deployment.py`는 삭제**한다. 다만 수신자 소스는 파일이 아니라 `config.py`의 기존 관례(env 기반)를 따라 `EMAIL_TO`를 쉼표로 구분한 다중 주소로 확장한다 — 저장소에 없는 JSON 픽스처 파일에 의존하지 않고, `.env` 하나로 계속 관리할 수 있다. `tools/email_send.py`의 `send_email()`은 지금 `config.EMAIL_TO`만 내부적으로 사용해 수신자를 무시하므로, 진짜 다중 발송이 되려면 `to` 파라미터를 받아야 한다(이게 없으면 그냥 같은 주소로 N번 보내는 것과 같다 — `deployment.py`의 원래 docstring도 이 문제를 알고 있었다).

- [ ] **Step 1: 실패하는 테스트부터 작성** — `tests/test_distributor.py` 신규 생성:

```python
# tests/test_distributor.py
import unittest
from unittest.mock import MagicMock

from agents.distributor import DistributorAgent


def _make_state(digest_id=1):
    return {
        "digest_date": "2026-08-18",
        "keyword": "AI",
        "email_html": "<p>본문</p>",
        "digest_id": digest_id,
    }


class DistributorMultiRecipientTest(unittest.TestCase):
    def test_수신자별로_개별_발송하고_개별_로그를_남긴다(self):
        email_tool = MagicMock(return_value={"id": "sent"})
        log_send_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": log_send_tool},
            channels=["email"],
        )

        with unittest.mock.patch("agents.distributor.config") as mock_config:
            mock_config.EMAIL_RECIPIENTS = ["a@example.com", "b@example.com"]
            result = agent.run(_make_state())

        self.assertEqual(email_tool.call_count, 2)
        email_tool.assert_any_call(unittest.mock.ANY, "<p>본문</p>", to="a@example.com")
        email_tool.assert_any_call(unittest.mock.ANY, "<p>본문</p>", to="b@example.com")
        self.assertEqual(log_send_tool.call_count, 2)
        self.assertEqual(result["send_result"]["success_count"], 2)
        self.assertEqual(result["send_result"]["total_count"], 2)

    def test_일부_수신자_발송_실패해도_나머지는_계속_보낸다(self):
        email_tool = MagicMock(side_effect=[RuntimeError("발송 실패"), {"id": "sent"}])
        log_send_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": log_send_tool},
            channels=["email"],
        )

        with unittest.mock.patch("agents.distributor.config") as mock_config:
            mock_config.EMAIL_RECIPIENTS = ["fail@example.com", "ok@example.com"]
            result = agent.run(_make_state())

        self.assertEqual(result["send_result"]["success_count"], 1)
        self.assertEqual(result["send_result"]["total_count"], 2)
        log_send_tool.assert_any_call(1, "email", "fail@example.com", "failed", "발송 실패")
        log_send_tool.assert_any_call(1, "email", "ok@example.com", "success")

    def test_email이_channels에_없으면_아무것도_하지_않는다(self):
        email_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": MagicMock()},
            channels=[],
        )
        result = agent.run(_make_state())
        email_tool.assert_not_called()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest tests.test_distributor -v
```

Expected: FAIL — `AttributeError`(email_tool이 `to=` 키워드를 모름) 또는 `config.EMAIL_RECIPIENTS` 없음 계열 에러.

- [ ] **Step 3: `config.py`에 다중 수신자 파싱 추가**

`config.py`의 `EMAIL_TO = os.getenv("EMAIL_TO", "")` 바로 아래에 추가:

```python
    EMAIL_TO = os.getenv("EMAIL_TO", "")
    # EMAIL_TO에 쉼표로 여러 주소를 넣으면 각각에 개별 발송한다 (agents/distributor.py).
    EMAIL_RECIPIENTS = [addr.strip() for addr in EMAIL_TO.split(",") if addr.strip()]
```

- [ ] **Step 4: `.env.example` 갱신**

`EMAIL_TO="you@example.com"` 줄 위에 주석 추가:

```
# 쉼표로 구분하면 여러 명에게 개별 발송한다 (예: "a@example.com,b@example.com")
EMAIL_TO="you@example.com"
```

- [ ] **Step 5: `tools/email_send.py`에 `to` 파라미터 추가**

`send_email` 함수를 다음으로 교체:

```python
def send_email(subject: str, html: str, to: str | None = None) -> dict:
    recipient = to or config.EMAIL_TO
    if not config.RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY가 설정되지 않았습니다 (.env 확인)")
    if not config.EMAIL_FROM or not recipient:
        raise RuntimeError("EMAIL_FROM / EMAIL_TO가 설정되지 않았습니다 (.env 확인)")

    resp = requests.post(
        RESEND_URL,
        headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"},
        json={
            "from": config.EMAIL_FROM,
            "to": [recipient],
            "subject": subject,
            "html": html,
        },
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Resend 발송 실패 ({resp.status_code}): {resp.text}")
    return resp.json()
```

- [ ] **Step 6: `agents/distributor.py` 다중 수신자 루프로 교체**

파일 전체를 다음으로 교체:

```python
"""채널별 발송을 담당하는 에이전트.

email 채널은 config.EMAIL_RECIPIENTS(EMAIL_TO를 쉼표로 나눈 목록)의 각 주소에 개별
발송하고, 성공/실패를 주소별로 log_send에 남긴다 — 구 tools/deployment.py가 하던
다중 수신자 배포 로직을 여기로 흡수했다(수신자 소스만 로컬 JSON 파일 대신
config.py의 기존 env 관례를 따르도록 바꿨다). Discord 발송(백로그 #9)을 추가할 때도
오케스트레이터나 다른 에이전트는 건드릴 필요 없이, registry.py의 tools에 "discord"를
등록하고 channels 목록에 이름만 추가하면 된다.
"""
from config import config

from .base import Agent


class DistributorAgent(Agent):
    def __init__(self, name: str, tools: dict, channels: list[str]):
        super().__init__(name=name, tools=tools)
        self.channels = channels

    def run(self, state: dict) -> dict:
        subject = f"[{state['digest_date']}] {state['keyword']} 뉴스·기술 다이제스트"

        if "email" not in self.channels or "email" not in self.tools:
            return {}

        recipients = config.EMAIL_RECIPIENTS or ([config.EMAIL_TO] if config.EMAIL_TO else [])
        results = []
        for recipient in recipients:
            try:
                result = self.tools["email"](subject, state["email_html"], to=recipient)
                self.tools["log_send"](state["digest_id"], "email", recipient, "success")
                results.append({"recipient": recipient, "status": "success", "result": result})
            except Exception as exc:  # noqa: BLE001 - MVP 단계는 넓게 잡고 로그만 남긴다
                self.tools["log_send"](state["digest_id"], "email", recipient, "failed", str(exc))
                results.append({"recipient": recipient, "status": "failed", "error": str(exc)})

        return {
            "send_result": {
                "recipients": results,
                "success_count": sum(r["status"] == "success" for r in results),
                "total_count": len(results),
            }
        }
```

- [ ] **Step 7: 고아 모듈 삭제**

```bash
git rm tools/deployment.py tests/test_deployment.py
```

- [ ] **Step 8: 새 테스트 통과 확인**

```bash
python -m unittest tests.test_distributor -v
```

Expected: 3개 전부 PASS.

- [ ] **Step 9: 회귀 확인 (DB 없이 도는 것들)**

```bash
python -m unittest tests.test_main_modes tests.test_scheduler tests.test_discord_send tests.test_distributor -v
```

Expected: 전부 PASS. `graph/pipeline.py`는 `agents["distributor"].run(state)`를 그대로 호출하므로 노드 코드 변경 없이 새 반환 형태(`send_result` 구조 변경)가 그대로 흘러간다 — `main.py:run_digest`의 로그 포맷 문자열(`"발송 결과 %s", result["send_result"]`)은 dict를 그대로 `%s`로 찍으므로 깨지지 않는다.

- [ ] **Step 10: Commit**

```bash
git add config.py .env.example tools/email_send.py agents/distributor.py tests/test_distributor.py
git commit -m "refactor: tools/deployment.py의 다중 수신자 발송 로직을 agents/distributor.py로 흡수하고 고아 모듈 제거"
```

---

## Task 8: 로컬 DB 컨테이너를 pgvector 이미지로 재생성

**Files:** 없음 (인프라 작업, 코드 변경 없음)

**배경:** `Dockerfile.db`는 이미 `pgvector/pgvector:pg16` 베이스로 올바르게 작성돼 있지만, 현재 로컬에서 떠 있는 `briefyi-db-1` 컨테이너(`briefyi-db:16` 이미지)는 그 수정 전에 빌드된 것이라 `\dx` 확인 결과 `vector` 확장이 없다. `db/db.py`의 `init_db()`는 `schema.sql`과 `vector_schema.sql`을 무조건 둘 다 적용하므로, 재빌드 전까지는 `main.py`/`rag` 계열 명령이 전부 `CREATE EXTENSION vector` 단계에서 실패한다.

- [ ] **Step 1: 현재 상태 재확인**

```bash
docker exec briefyi-db-1 psql -U briefyi -d briefyi -c "\dx"
```

Expected: `plpgsql`만 나오고 `vector`는 없음.

- [ ] **Step 2: 볼륨을 포함해 재생성** (기존 로컬 데이터는 테스트/개발용이므로 삭제해도 무방 — 운영 데이터가 이 볼륨에 있다면 이 스텝 전에 먼저 백업할 것)

```bash
./scripts/db-up.sh --reset --force
```

내부적으로 `docker compose build db && docker compose down -v && docker compose up -d db`에 준하는 동작을 한다(스크립트 정확한 동작은 `scripts/db-up.sh` 참고). Windows라면 `./scripts/db-up.ps1 -Reset -Force`.

- [ ] **Step 3: pgvector 확장이 이제 있는지 확인**

```bash
docker exec briefyi-db-1 psql -U briefyi -d briefyi -c "\dx"
```

Expected: `plpgsql`과 `vector` 둘 다 나옴.

- [ ] **Step 4: DB 관련 테스트 전체 재실행**

```bash
python -m unittest tests.test_db tests.test_db_connection tests.test_db_crud -v
```

Expected: 이전에 `errors=12`였던 것이 전부 PASS로 바뀜(단, `pgvector` 파이썬 패키지가 이 venv에 없다면 `rag/tests`의 4개 모듈은 여전히 import 에러 — 이건 별도 이슈이며 이 태스크의 범위가 아니다. 필요하면 `uv pip install pgvector beautifulsoup4`로 추가 설치).

- [ ] **Step 5: 커밋할 코드 변경 없음 — 대신 상태 기록**

이 태스크는 로컬 인프라 상태 변경이라 커밋할 파일이 없다. `docs/project-status-2026-08.md`의 §2.4 상태 라인만 필요하면 사람이 직접 갱신한다(자동화하지 않음 — 상태 문서는 스냅샷이므로 다음 전체 점검 때 다시 씀).

---

## Task 9: GitHub Actions `daily-digest.yml`을 pgvector 이미지로 교체

**Files:**
- Modify: `.github/workflows/daily-digest.yml`

**배경:** `services.postgres`가 `postgres:16-alpine`(pgvector 미포함)을 쓰고 있어, `main.py`의 `init_db()`가 `vector_schema.sql`을 적용하는 순간 매 실행이 실패한다. `Dockerfile.db`가 쓰는 것과 같은 `pgvector/pgvector:pg16` 이미지로 바꾼다.

- [ ] **Step 1: 현재 설정 확인**

```bash
grep -n "image: postgres" .github/workflows/daily-digest.yml
```

Expected: `image: postgres:16-alpine`.

- [ ] **Step 2: 이미지 교체**

`.github/workflows/daily-digest.yml`의 `services.postgres.image` 값을 변경:

```yaml
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_DB: briefyi
          POSTGRES_USER: briefyi
          POSTGRES_PASSWORD: briefyi
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U briefyi -d briefyi"
          --health-interval 5s
          --health-timeout 5s
          --health-retries 10
```

(`env`/`ports`/`options`는 그대로, `image` 한 줄만 변경)

- [ ] **Step 3: YAML 문법 확인**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/daily-digest.yml'))" && echo "YAML OK"
```

Expected: `YAML OK` 출력, 예외 없음.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/daily-digest.yml
git commit -m "fix: daily-digest 워크플로가 pgvector 없는 postgres:16-alpine을 써서 init_db()가 매번 실패하던 문제 수정"
```

- [ ] **Step 5: 실제 검증은 push 후 GitHub Actions에서** — `workflow_dispatch`로 수동 실행하거나 다음 스케줄(매일 08:00 KST)을 기다려 로그에서 `CREATE EXTENSION vector` 단계 성공을 직접 확인한다(로컬에서는 재현 불가, 이 스텝은 계획 문서상 기록만 남기고 실제 확인은 별도로 진행).

---

## Task 10: `rag_latest/` 신설 — `rag/` 기반 + `rag_experiment/`의 유용한 기능 통합

**배경 (조사 결과 요약):** `rag/`(프로덕션 트랙)와 `rag_experiment/`(초기 실험, `rag_documents`라는 별도 flat 테이블과 `psycopg2`/로컬 임베딩 모델을 씀 — 지금 스키마·드라이버와 다름)를 둘 다 보존한 채, 새 `rag_latest/`에 `rag/`를 뼈대로 삼고 `rag_experiment/`에만 있는 두 가지 실질적 가치를 이식한다: **cross-encoder reranking**과 **LLM tool-use 검색 래퍼**. `rag_experiment/rag/classify.py`(Claude 기반 5-카테고리 분류)는 `rag/taxonomy.py`(GLiNER2 기반, 버전 관리됨, 로컬이라 API 비용 없음)가 이미 더 나은 버전을 갖고 있으므로 포팅하지 않는다. `rag_experiment/rag/db_config.py`·`migrate_schema.py`·`indexer.py`·`retriever.py`(flat 스키마 대상)도 포팅하지 않는다 — `rag/`의 `db.py`/`retriever.py`가 정규화된 스키마와 hybrid RRF + soft boosting으로 이미 더 우수하다. `sbs_fetch.py`/`sbs_to_rag.py`/`bulk_collect.py`/`news_preprocess.py`/`newsletter.py`/`query_rag.py`/`summarizer.py`도 포팅하지 않는다 — SBS 전용 수집기이거나(뉴스 소스 자체는 이 계획 범위 밖), summarizer.py는 실제로는 500자 truncation일 뿐인 스텁이다.

**Files:**
- Create: `rag_latest/` (전체 새 디렉터리, `rag/`를 복사한 뒤 수정)
- Create: `rag_latest/reranker.py`
- Create: `rag_latest/agent_tool.py`
- Create: `rag_latest/eval.py`
- Create: `rag_latest/requirements-rerank.txt`
- Create: `rag_latest/tests/test_reranker.py`, `rag_latest/tests/test_agent_tool.py`, `rag_latest/tests/test_eval.py`
- 수정 없음: `rag/`, `rag_experiment/` (읽기 전용 참고 소스, Global Constraints 참고)

**Interfaces:**
- Consumes: `rag/`의 기존 공개 API 전체 (`retriever.retrieve()`, `db.py`의 함수들, `taxonomy.CATEGORY_LABELS`/`DOMAIN_LABELS`) — 복사 후 `rag_latest` 네임스페이스에서 동일한 시그니처로 그대로 사용
- Produces: `rag_latest.reranker.rerank(query, rows, top_k=None) -> list[dict]`, `rag_latest.agent_tool.search_news(query, top_k=5, category=None, domains=None) -> list[dict]`, `rag_latest.agent_tool.SEARCH_NEWS_TOOL_SCHEMA: dict`, `rag_latest.eval.evaluate_self_retrieval(...) -> dict`

### Task 10a: `rag_latest/` 스캐폴드 — `rag/`를 뼈대로 복사

**Files:**
- Create: `rag_latest/` (rag/ 전체 복사: `content.py`, `embed.py`, `extract.py`, `taxonomy.py`, `db.py`, `indexer.py`, `retriever.py`, `pipeline.py`, `cli.py`, `__init__.py`, `README.md`, `requirements-event.txt`, `tests/`)

**배경:** `rag/*.py`는 전부 상대 import(`from . import db`, `from .embed import embed_query`)만 쓰므로 그대로 복사해도 내부 참조는 깨지지 않는다. 유일하게 손볼 곳은 `rag/tests/*.py`가 쓰는 절대 import(`from rag.xxx import ...`, `from rag import db`, `from rag.tests.test_content import OffsetTokenizer`)다 — `rag_latest`로 바꿔야 한다.

- [ ] **Step 1: 복사**

```bash
cd /home/ysoh1113/workspace/projects/BrieFYI
cp -r rag rag_latest
```

- [ ] **Step 2: 테스트 파일의 절대 import를 rag_latest로 치환**

```bash
sed -i 's/from rag\./from rag_latest./g; s/from rag import/from rag_latest import/g' rag_latest/tests/*.py
grep -rn "^from rag\b\|^import rag\b" rag_latest/tests/*.py
```

Expected: 마지막 grep 결과 없음(전부 `rag_latest.`로 치환됨), `from rag_latest.tests.test_content import OffsetTokenizer` 같은 형태로 남아있는지 확인.

- [ ] **Step 3: `rag_latest/__init__.py` 설명 갱신**

`rag_latest/__init__.py`의 docstring 첫 줄과 구성 설명 아래에 한 줄 추가:

```python
"""기사 본문 청킹, embedding, GLiNER2 4-Layer indexing, retrieval을 제공하는 RAG 패키지.

rag/를 뼈대로 하고, rag_experiment/에서 검증된 reranker.py(cross-encoder 재정렬)와
agent_tool.py(LLM tool-use 검색 래퍼)를 이식했다. rag/, rag_experiment/는 원본 보존을
위해 그대로 두고 손대지 않는다(추후 삭제 예정).

... (기존 파일 목록 설명 유지) ...

    reranker.py   cross-encoder 재정렬 (rag_experiment/rag/reranker.py 이식)
    agent_tool.py LLM tool-use 검색 래퍼 (rag_experiment/rag/agent_search.py 이식)
    eval.py       recall@k/MRR 평가 (rag_experiment의 eval_*.py 계열 이식/통합)
"""

__version__ = "0.1.0"
```

- [ ] **Step 4: 패키지 임포트 확인**

```bash
.venv/bin/python -c "import rag_latest; from rag_latest import retriever, db, taxonomy; print('OK')"
```

Expected: `OK` 출력, ImportError 없음.

- [ ] **Step 5: 테스트 parity 확인 (rag/tests와 같은 결과가 나와야 정상)**

```bash
python -m unittest discover -s rag/tests -t . -p "test_*.py" 2>&1 | tail -5
python -m unittest discover -s rag_latest/tests -t . -p "test_*.py" 2>&1 | tail -5
```

Expected: 두 결과가 동일해야 한다(현재 `pgvector`/`beautifulsoup4` 파이썬 패키지가 venv에 없어 `rag/tests`도 일부 ERROR/skip이 있다면, `rag_latest/tests`도 정확히 같은 개수의 ERROR/skip이 나오는 게 정상 — 이 태스크에서 새로 깨진 게 없다는 뜻이다. 새로운 ERROR가 늘었다면 sed 치환이 덜 된 파일이 있는지 다시 확인).

- [ ] **Step 6: Commit**

```bash
git add rag_latest/
git commit -m "feat: rag/를 뼈대로 rag_latest/ 스캐폴드 생성 (내부 로직 변경 없음, 테스트 import만 rag_latest로 조정)"
```

### Task 10b: cross-encoder reranker 이식

**Files:**
- Create: `rag_latest/reranker.py`
- Create: `rag_latest/requirements-rerank.txt`
- Test: `rag_latest/tests/test_reranker.py`

**Interfaces:**
- Consumes: `rag_latest.retriever.retrieve()`가 반환하는 행 형태(`chunk_id`, `title`, `text`, `score` 등 키를 가진 `dict`) — Task 10a에서 그대로 복사됐으므로 `rag/retriever.py:33-60`의 `_merge_candidates`가 만드는 키 구성과 동일
- Produces: `rerank(query: str, rows: list[dict], top_k: int | None = None) -> list[dict]` — 각 행에 `pre_rerank_rank`, `rerank_score` 키 추가, `rerank_score` 내림차순 정렬

**배경:** `rag_experiment/rag/reranker.py`의 `BAAI/bge-reranker-v2-m3` 접근을 그대로 가져오되, 입력 행 형태를 `rag_experiment`의 flat `title`/`content` 키가 아니라 `rag/retriever.py`가 실제로 반환하는 `title`/`text` 키에 맞춘다. `rag/retriever.py`(→ `rag_latest/retriever.py`)에는 이 재정렬 단계가 없으므로, 검색 자체는 건드리지 않고 검색 결과를 후처리하는 별도 함수로 둔다(호출은 Task 10c의 `agent_tool.py`에서 한다 — `retriever.py` 자체를 수정하지 않아 Task 10a에서 확인한 parity가 계속 유지된다).

- [ ] **Step 1: 재정렬 로직의 실패 테스트부터 작성** — `rag_latest/tests/test_reranker.py`:

```python
import unittest
from unittest.mock import patch

from rag_latest.reranker import rerank


class RerankTest(unittest.TestCase):
    def test_후보를_cross_encoder_점수_순으로_재정렬한다(self):
        rows = [
            {"chunk_id": 1, "title": "A", "text": "text a", "score": 0.9},
            {"chunk_id": 2, "title": "B", "text": "text b", "score": 0.8},
        ]
        fake_reranker = type(
            "FakeReranker", (), {"compute_score": staticmethod(lambda pairs, normalize=True: [0.1, 0.9])}
        )()
        with patch("rag_latest.reranker.get_reranker", return_value=fake_reranker):
            result = rerank("query", rows)

        self.assertEqual([row["chunk_id"] for row in result], [2, 1])
        self.assertAlmostEqual(result[0]["rerank_score"], 0.9)
        self.assertEqual(result[0]["pre_rerank_rank"], 2)

    def test_빈_후보는_그대로_빈_리스트를_반환한다(self):
        self.assertEqual(rerank("query", []), [])

    def test_top_k로_결과를_자른다(self):
        rows = [{"chunk_id": i, "title": "t", "text": "x", "score": 0.0} for i in range(5)]
        fake_reranker = type(
            "FakeReranker", (), {"compute_score": staticmethod(lambda pairs, normalize=True: [0.5] * 5)}
        )()
        with patch("rag_latest.reranker.get_reranker", return_value=fake_reranker):
            result = rerank("query", rows, top_k=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest rag_latest.tests.test_reranker -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag_latest.reranker'`.

- [ ] **Step 3: `rag_latest/reranker.py` 작성**

```python
"""검색 결과 재순위화 (cross-encoder reranker).

rag_experiment/rag/reranker.py의 BAAI/bge-reranker-v2-m3 접근을 rag_latest.retriever의
retrieve() 출력 행 형태(title/text 키)에 맞춰 재작성했다. retrieve()가 반환한 후보를
받아 재정렬만 하고, 검색 자체(vector/text/hybrid)는 건드리지 않는다.

로컬 모델을 GPU/CPU에 올리므로 embed.py(HF Inference API, 로컬 메모리 사용 안 함)와
달리 별도 프로세스 메모리를 쓴다. reranker와 embedding 모델을 동시에 올리지 않도록
호출 측(agent_tool.py)이 unload_reranker()로 정리한다.
"""
from __future__ import annotations

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    return _reranker


def unload_reranker() -> None:
    global _reranker
    if _reranker is not None:
        del _reranker
        _reranker = None
        import gc

        gc.collect()


def rerank(query: str, rows: list[dict], top_k: int | None = None) -> list[dict]:
    """retrieve()가 반환한 후보 목록을 cross-encoder 점수로 재정렬한다.

    row["score"](RRF/boost 점수)는 건드리지 않고 row["rerank_score"]를 새로 추가한 뒤
    그 값으로 재정렬한다 — 재정렬 전 순위(row["pre_rerank_rank"])도 함께 남겨 비교할 수
    있게 한다.
    """
    if not rows:
        return []

    reranker = get_reranker()
    pairs = [(query, f"{row['title']}\n{row['text']}") for row in rows]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    reranked = [
        {**row, "pre_rerank_rank": i + 1, "rerank_score": float(score)}
        for i, (row, score) in enumerate(zip(rows, scores))
    ]
    reranked.sort(key=lambda row: (-row["rerank_score"], row["chunk_id"]))
    return reranked[:top_k] if top_k is not None else reranked
```

- [ ] **Step 4: 로컬 전용 의존성 파일 작성** — `rag_latest/requirements-rerank.txt` (기존 `rag/requirements-event.txt` 관례를 따름, 공유 `requirements.txt`에는 넣지 않음):

```
# Local cross-encoder reranker 전용. 공유 requirements.txt에는 포함하지 않는다.
FlagEmbedding>=1.2.0
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m unittest rag_latest.tests.test_reranker -v
```

Expected: 3개 전부 PASS(모두 `get_reranker`를 mock하므로 실제 `FlagEmbedding` 설치 없이도 통과).

- [ ] **Step 6: Commit**

```bash
git add rag_latest/reranker.py rag_latest/requirements-rerank.txt rag_latest/tests/test_reranker.py
git commit -m "feat(rag_latest): rag_experiment의 cross-encoder reranker를 retrieve() 출력 형태에 맞춰 이식"
```

### Task 10c: LLM tool-use 검색 래퍼 이식

**Files:**
- Create: `rag_latest/agent_tool.py`
- Test: `rag_latest/tests/test_agent_tool.py`

**Interfaces:**
- Consumes: `rag_latest.retriever.retrieve(...)`, `rag_latest.reranker.rerank(...)`/`unload_reranker()`(Task 10b), `rag_latest.taxonomy.CATEGORY_LABELS`/`DOMAIN_LABELS`
- Produces: `SEARCH_NEWS_TOOL_SCHEMA: dict`(Anthropic tool-use 스키마), `search_news(query, top_k=5, category=None, domains=None) -> list[dict]`, `execute_search_news(arguments: dict) -> list[dict]`

**배경:** `rag_experiment/rag/agent_search.py` + `agent_tool_schema.py`를 이식하되, 두 가지를 바꾼다. (1) 검색 자체를 `rag_experiment`의 flat-table 필터 쿼리 대신 `rag_latest.retriever.retrieve(search_mode="hybrid", ...)`(vector+text+RRF+boosting)로 교체한다. (2) tool schema의 `category`/`domain` enum 목록을 `classify.py`의 자체 하드코딩 대신 `rag_latest.taxonomy.CATEGORY_LABELS`/`DOMAIN_LABELS`(GLiNER2 기반, 실제 DB에 저장되는 값과 일치)에서 가져온다 — 두 taxonomy는 값이 다르므로(예: `classify.py`는 5개 카테고리, `taxonomy.py`도 5개지만 도메인 개수가 다름) 실제 저장된 값과 다른 enum을 스키마에 노출하면 필터가 항상 빈 결과를 낼 위험이 있다.

- [ ] **Step 1: 실패하는 테스트부터 작성** — `rag_latest/tests/test_agent_tool.py`:

```python
import unittest
from unittest.mock import patch

from rag_latest.agent_tool import SEARCH_NEWS_TOOL_SCHEMA, execute_search_news, search_news


class SearchNewsTest(unittest.TestCase):
    def test_retrieve와_rerank를_거쳐_직렬화된_결과를_반환한다(self):
        fake_rows = [
            {
                "article_id": 1,
                "chunk_id": 10,
                "title": "제목",
                "url": "https://example.com/1",
                "text": "x" * 600,
                "category": "기술",
                "domains": ["AI"],
                "score": 0.5,
            }
        ]
        reranked_rows = [{**fake_rows[0], "rerank_score": 0.987654}]

        with patch("rag_latest.agent_tool.retriever.retrieve", return_value=fake_rows) as mock_retrieve, \
             patch("rag_latest.agent_tool.reranker.rerank", return_value=reranked_rows) as mock_rerank, \
             patch("rag_latest.agent_tool.reranker.unload_reranker") as mock_unload:
            result = search_news("Claude 관련 뉴스", top_k=5)

        mock_retrieve.assert_called_once()
        mock_rerank.assert_called_once()
        mock_unload.assert_called_once()
        self.assertEqual(result[0]["article_id"], 1)
        self.assertEqual(len(result[0]["text"]), 500)
        self.assertEqual(result[0]["rerank_score"], 0.9877)

    def test_execute_search_news는_dict_인자를_search_news로_그대로_전달한다(self):
        with patch("rag_latest.agent_tool.search_news", return_value=[]) as mock_search:
            execute_search_news({"query": "AI", "top_k": 3, "category": "기술"})

        mock_search.assert_called_once_with(query="AI", top_k=3, category="기술", domains=None)

    def test_tool_schema의_category_enum은_taxonomy와_일치한다(self):
        from rag_latest import taxonomy

        enum_values = SEARCH_NEWS_TOOL_SCHEMA["input_schema"]["properties"]["category"]["enum"]
        self.assertEqual(enum_values, list(taxonomy.CATEGORY_LABELS))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest rag_latest.tests.test_agent_tool -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag_latest.agent_tool'`.

- [ ] **Step 3: `rag_latest/agent_tool.py` 작성**

```python
"""LLM tool-use용 뉴스 검색 도구.

rag_experiment/rag/agent_search.py + agent_tool_schema.py를 rag_latest.retriever.retrieve()와
rag_latest.taxonomy의 실제 CATEGORY_LABELS/DOMAIN_LABELS에 맞춰 재작성했다.
rag_experiment/rag/classify.py의 자체 taxonomy는 쓰지 않는다 — DB에 실제로 저장되는
값(GLiNER2 기반 taxonomy.py)과 다른 enum을 노출하면 필터가 항상 빈 결과를 낼 수 있다.
"""
from __future__ import annotations

from typing import Any

from . import reranker, retriever, taxonomy

SEARCH_NEWS_TOOL_SCHEMA = {
    "name": "search_news",
    "description": "저장된 뉴스 기사에서 질의와 관련된 chunk를 검색한다. category/domain으로 좁혀 검색할 수 있다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "검색할 자연어 질의"},
            "top_k": {"type": "integer", "description": "반환할 결과 개수", "default": 5},
            "category": {
                "type": "string",
                "enum": list(taxonomy.CATEGORY_LABELS),
                "description": "결과를 좁힐 대분류(선택)",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string", "enum": list(taxonomy.DOMAIN_LABELS)},
                "description": "결과를 좁힐 세부 도메인(선택, 복수 가능)",
            },
        },
        "required": ["query"],
    },
}


def search_news(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """검색 -> rerank까지 수행하고 LLM에 돌려줄 수 있게 직렬화된 결과를 반환한다."""
    candidates = retriever.retrieve(
        query=query,
        top_k=max(top_k * 3, top_k),
        search_mode="hybrid",
        category=category,
        domains=domains,
    )
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    reranker.unload_reranker()

    return [
        {
            "article_id": row["article_id"],
            "title": row["title"],
            "url": row["url"],
            "text": row["text"][:500],
            "category": row.get("category"),
            "domains": row.get("domains") or [],
            "rerank_score": round(row["rerank_score"], 4),
        }
        for row in reranked
    ]


def execute_search_news(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Anthropic tool-use 루프에서 SEARCH_NEWS_TOOL_SCHEMA 호출 결과를 그대로 실행하는 dispatch shim."""
    return search_news(
        query=arguments["query"],
        top_k=arguments.get("top_k", 5),
        category=arguments.get("category"),
        domains=arguments.get("domains"),
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m unittest rag_latest.tests.test_agent_tool -v
```

Expected: 3개 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add rag_latest/agent_tool.py rag_latest/tests/test_agent_tool.py
git commit -m "feat(rag_latest): rag_experiment의 agent tool-use 검색 래퍼를 retrieve()+실제 taxonomy 기준으로 이식"
```

### Task 10d: 평가(recall@k / MRR) 하네스 이식

**Files:**
- Create: `rag_latest/eval.py`
- Test: `rag_latest/tests/test_eval.py`

**Interfaces:**
- Consumes: `rag_latest.db`, `rag_latest.retriever.retrieve()`
- Produces: `evaluate_self_retrieval(article_ids=None, top_k_values=(5, 10)) -> dict`(각 기사 제목으로 자기 자신을 검색했을 때의 Recall@5/@10/MRR), 순수 채점 함수 `reciprocal_rank(ranked_ids, target_id) -> float`, `recall_at_k(ranked_ids, target_id, k) -> bool`

**배경:** `rag_experiment`에는 평가 스크립트가 7개(`eval_dataset.py`, `eval_gold.py`, `eval_all_report.py`, `eval_compare.py`, `eval_compare_noisy.py`, `eval_live_sbs.py`, `gold_dataset_gen.py`) 있었지만, 방법론은 근본적으로 두 갈래다 — (a) 합성 데이터로 알려진 정답을 검증, (b) 실제 색인된 기사에 대해 자기 자신의 제목으로 검색해 자기 chunk가 몇 위에 나오는지 측정(self-retrieval). `recall_result.md`/`gold_eval_result.md`의 핵심 결론은 **"뚜렷이 구별되는 기사는 recall이 거의 항상 1.0에 가깝지만, 근접 중복 기사(같은 사건을 다룬 여러 기사)가 있으면 순위가 밀린다"**는 것이었다 — 이건 별도의 "노이즈 비교" 스크립트 없이도 실제 DB 전체로 self-retrieval을 돌리면 자연히 드러나는 패턴이므로, `eval_compare_noisy.py`를 별도로 포팅하지 않고 (b) 방법론 하나로 합친다. 합성 데이터셋(a)과 LLM 기반 골드셋 생성(`gold_dataset_gen.py`)은 실제 DB에 유의미한 양의 기사가 쌓인 뒤에야 쓸모가 있으므로 이번 이식 범위에서는 제외한다(YAGNI — 지금 당장 필요하지 않은 인프라를 먼저 만들지 않는다). 점수 계산 함수(`reciprocal_rank`, `recall_at_k`)는 I/O 없이 순수 함수라 단위 테스트가 가능하지만, `evaluate_self_retrieval` 자체는 실제 DB + HF embedding API가 필요해 CI에서 자동 실행하지 않는 수동 리포트 도구다(원본 `rag_experiment`의 eval 스크립트들도 전부 그런 성격이었다).

- [ ] **Step 1: 순수 채점 함수의 실패 테스트부터 작성** — `rag_latest/tests/test_eval.py`:

```python
import unittest

from rag_latest.eval import recall_at_k, reciprocal_rank


class ScoringTest(unittest.TestCase):
    def test_reciprocal_rank는_target이_있는_위치의_역수를_반환한다(self):
        self.assertEqual(reciprocal_rank([10, 20, 30], 20), 0.5)
        self.assertEqual(reciprocal_rank([10, 20, 30], 10), 1.0)

    def test_reciprocal_rank는_target이_없으면_0을_반환한다(self):
        self.assertEqual(reciprocal_rank([10, 20, 30], 999), 0.0)

    def test_recall_at_k는_상위_k_안에_있는지_판정한다(self):
        self.assertTrue(recall_at_k([10, 20, 30, 40], 30, k=3))
        self.assertFalse(recall_at_k([10, 20, 30, 40], 40, k=3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest rag_latest.tests.test_eval -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag_latest.eval'`.

- [ ] **Step 3: `rag_latest/eval.py` 작성**

```python
"""검색 품질 평가 (Recall@k / MRR).

rag_experiment의 eval_*.py 계열(7개 스크립트)을 self-retrieval 방법론 하나로 통합해
rag_latest.retriever.retrieve()를 대상으로 재작성했다. 합성 데이터셋·LLM 골드셋 생성은
실제 색인 데이터가 충분히 쌓인 뒤에야 의미가 있어 이번 이식 범위에서 제외했다
(rag_experiment/gold_eval_result.md, recall_result.md 참고).

DB + HF embedding API가 필요한 수동 리포트 도구다. CI에서 자동 실행하지 않는다.

    python -m rag_latest.eval self-retrieval --top-k 5 10
"""
from __future__ import annotations

from . import db, retriever


def reciprocal_rank(ranked_ids: list[int], target_id: int) -> float:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id == target_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[int], target_id: int, k: int) -> bool:
    return target_id in ranked_ids[:k]


def evaluate_self_retrieval(
    article_ids: list[int] | None = None,
    top_k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """각 기사의 제목으로 검색했을 때 그 기사 자신의 chunk가 몇 위에 나오는지 측정한다.

    recall_result.md의 Part B(231개 실제 기사, Recall@5=0.961/Recall@10=0.991/MRR=0.844)와
    같은 방법론이다 — "뚜렷이 구별되는 기사는 거의 항상 상위에 나오지만, 같은 사건을 다룬
    근접 중복 기사가 많을수록 순위가 밀린다"는 패턴이 여기서도 드러나는지 확인하는 용도다.
    """
    if article_ids is None:
        # "인덱싱된 기사 전부"를 자동으로 고르는 쿼리는 db.py에 없다(색인 안 된 기사를
        # 찾는 load_unindexed_article_ids()의 반대 개념). 새 쿼리를 추가하는 대신, 평가
        # 대상은 호출자가 이미 index_articles()로 색인한 ID 목록을 명시적으로 넘기게 한다.
        raise ValueError(
            "article_ids를 명시적으로 넘겨야 한다 (예: 이미 index_articles()로 색인된 ID 목록)."
        )

    articles = {row["id"]: row for row in db.load_articles(article_ids)}
    max_k = max(top_k_values)

    reciprocal_ranks = []
    hits_by_k = {k: 0 for k in top_k_values}
    evaluated = 0

    for article_id in article_ids:
        article = articles.get(article_id)
        if article is None or not article.get("title"):
            continue
        rows = retriever.retrieve(article["title"], top_k=max_k, search_mode="hybrid")
        ranked_article_ids = [row["article_id"] for row in rows]

        evaluated += 1
        reciprocal_ranks.append(reciprocal_rank(ranked_article_ids, article_id))
        for k in top_k_values:
            if recall_at_k(ranked_article_ids, article_id, k):
                hits_by_k[k] += 1

    if evaluated == 0:
        return {"evaluated": 0, "mrr": 0.0, "recall_at_k": {k: 0.0 for k in top_k_values}}

    return {
        "evaluated": evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
        "recall_at_k": {k: hits_by_k[k] / evaluated for k in top_k_values},
    }
```

*(참고: `evaluate_self_retrieval`은 호출자가 평가 대상 `article_ids`를 명시적으로 넘기는 걸 기본 계약으로 삼는다 — "DB의 모든 기사"를 자동으로 고르는 건 이 함수의 책임이 아니라 CLI/호출 스크립트 쪽 책임으로 남겨, `db.py`에 새 쿼리 함수를 추가하지 않고 기존 `load_articles`만 재사용한다.)*

- [ ] **Step 4: 순수 함수 테스트 통과 확인**

```bash
python -m unittest rag_latest.tests.test_eval -v
```

Expected: 3개 전부 PASS.

- [ ] **Step 5: (선택, DB 필요) 실제 색인된 기사가 있다면 수동 스모크 실행**

```bash
.venv/bin/python -c "
from rag_latest.eval import evaluate_self_retrieval
from rag_latest import db
ids = db.load_article_ids_by_urls([])  # 실제로는 이미 색인된 article_id 목록을 넣는다
print(evaluate_self_retrieval(article_ids=ids, top_k_values=(5, 10)))
"
```

Expected: `HF_TOKEN`/DB 접속이 준비돼 있고 색인된 기사가 있으면 `{"evaluated": N, "mrr": ..., "recall_at_k": {5: ..., 10: ...}}` 형태 출력. 색인된 기사가 아직 없다면 이 스텝은 건너뛰고 Step 4의 단위 테스트 통과만으로 이 태스크를 완료로 본다.

- [ ] **Step 6: Commit**

```bash
git add rag_latest/eval.py rag_latest/tests/test_eval.py
git commit -m "feat(rag_latest): rag_experiment eval 스크립트 7개를 self-retrieval Recall@k/MRR 하나로 통합 이식"
```

---

## 실행 후 전체 회귀 확인 (모든 태스크 완료 후)

```bash
python -m unittest discover -t .                                    # 루트 + rag/tests (rag/는 안 건드렸으므로 기존과 동일해야 함)
python -m unittest discover -s rag_latest/tests -t . -p "test_*.py"  # 신규
.venv/bin/python -m pytest data_pipeline/tests -q                    # 전부 통과해야 함 (기존 1건 실패 -> 0건)
.venv/bin/python -m pytest finetune/tests -q                         # 전부 통과해야 함 (기존 1건 실패 -> 0건)
```

**주의:** `data_pipeline/tests`와 `finetune/tests`를 **같은 pytest 명령에 함께** 넘기지 않는다 — 두
디렉터리 모두 `tests`라는 동일한 이름의 테스트 패키지를 갖고 있어, 한 번의 pytest 실행에 같이
넘기면 `ModuleNotFoundError: No module named 'tests.test_xxx'`류의 모듈 이름 충돌로 수집 자체가
실패한다(각 프로젝트의 `pyproject.toml`이 별도 rootdir로 실행될 때만 올바르게 동작). 항상 위처럼
디렉터리별로 따로 실행한다.

Expected: 이 계획 시작 시점 대비 새로 깨진 테스트가 없고(§project-status 문서의 `errors=12`가 pgvector 재빌드로 해소됐다면 0), `finetune`/`data_pipeline`은 각각 100% 통과로 바뀌어 있어야 한다.
