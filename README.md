# news-insight-agent (MVP 고정 파이프라인)

GNews 수집 → SQLite 저장 → 요약 → 인사이트/비즈니스 시사점 → 이메일 발송까지, 조건 분기 없이 고정된 순서로 실행되는 LangGraph 파이프라인이다. `mvp-implementation-breakdown.md`의 1단계(#1~#8) 구현체다.

## 구조

```
news-insight-agent/
  config.py              # .env 로드
  main.py                 # CLI 진입점 (#7 실행 트리거, #8 스케줄 대상)
  Dockerfile.db           # PostgreSQL 이미지 (스키마 초기화 포함)
  scripts/
    db-up.ps1 / db-up.sh  # DB 컨테이너 기동 스크립트
  db/
    schema.sql            # 테이블 정의 (PostgreSQL)
    db.py                  # PostgreSQL 헬퍼 (#2)
  tools/
    news_fetch.py          # GNews 수집 (#1)
    llm_client.py           # Anthropic 공용 호출
    summarize.py            # 요약 (#3)
    insight.py               # 인사이트+시사점 (#4)
    email_format.py          # Jinja2 렌더링 (#5)
    email_send.py            # Resend 발송 (#6)
  templates/
    digest_email.html.j2     # 이메일 템플릿
  graph/
    pipeline.py               # LangGraph StateGraph (#7)
  trigger/
    scheduler.py              # 간격 기반 트리거 스케줄러 (설계문서 3.1)
    jobs.py                   # 트리거가 실행할 작업 (현재는 hello_world)
    __main__.py               # 트리거 CLI
  tests/
    test_scheduler.py         # 트리거 테스트
  .github/workflows/
    daily-digest.yml          # 스케줄 실행 예시 (#8)
```

## 설정

1. `.env`에 `GNEWS_API_KEY`, `ANTHROPIC_API_KEY`, `RESEND_API_KEY`, `EMAIL_FROM`, `EMAIL_TO`를 채운다.
2. `pip install -r requirements.txt`
3. DB(PostgreSQL) 컨테이너를 띄운다.

```powershell
.\scripts\db-up.ps1          # Windows
```

```bash
./scripts/db-up.sh           # macOS / Linux
```

## DB (PostgreSQL 컨테이너)

기사·다이제스트·발송 이력은 `docker-compose.yml`의 `db` 서비스(PostgreSQL 16)에 저장한다.
이미지는 `Dockerfile.db`가 공식 `postgres:16-alpine`에 `db/schema.sql`을
`/docker-entrypoint-initdb.d`로 얹은 것이고, 데이터는 `pgdata` 볼륨에 영속화된다.

```powershell
.\scripts\db-up.ps1                # 빌드 + 기동 + 준비 대기 + 접속 정보 출력
.\scripts\db-up.ps1 -Psql          # 기동 후 psql 세션
.\scripts\db-up.ps1 -Logs          # 기동 후 로그 팔로우
.\scripts\db-up.ps1 -Down          # 정지 (데이터 유지)
.\scripts\db-up.ps1 -Reset -Force  # 데이터 볼륨까지 삭제하고 재생성
```

`db-up.sh`도 같은 기능을 `--psql`, `--logs`, `--down`, `--reset --force`로 제공한다.

접속 정보는 `DATABASE_URL` 하나로 결정된다. 없으면 `POSTGRES_HOST/PORT/DB/USER/PASSWORD`로
조립하며 기본값은 `postgresql://briefyi:briefyi@localhost:5432/briefyi`다. 컨테이너 안에서
실행할 때는 호스트가 `localhost`가 아니라 서비스명 `db`이며, 이 값은 `docker-compose.yml`이
주입한다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DATABASE_URL` | 아래 값들로 조립 | 접속 문자열. 관리형 DB를 쓸 때 이 값만 주면 된다 |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `5432` | 호스트에서 접속할 주소 |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | `briefyi` | DB 이름·계정. 컨테이너 초기화에도 같은 값이 쓰인다 |

스키마는 컨테이너 최초 기동 시 자동 적용되고, 앱 시작 시 `init_db()`가 `CREATE TABLE IF NOT
EXISTS`로 한 번 더 확인한다. 기존 볼륨이 있는 상태에서 스키마를 바꿨다면 `-Reset -Force`로
볼륨을 지우거나 마이그레이션을 직접 적용해야 한다.

계정 정보, 테이블별 컬럼/제약, 자주 쓰는 쿼리, 트러블슈팅은 `docs/database.md`에 정리해 두었다.

## 실행

두 가지 모드가 있다. 기본은 `single`이며 `RUN_MODE` 환경변수로 기본값을 바꿀 수 있다.

```bash
# single: 1회 실행 후 종료 (cron/GitHub Actions 등 외부 스케줄러가 정기 호출)
python main.py --keyword "AI" --days 1 --max-results 10

# trigger: 프로세스를 띄워둔 채 주기 반복 실행 (Ctrl+C 종료)
python main.py --mode trigger --interval 3600
python main.py --mode trigger --interval 10 --duration 60   # 60초만 돌고 종료
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--mode` | `RUN_MODE` (single) | `single`=1회 실행, `trigger`=주기 반복 |
| `--interval` | `TRIGGER_INTERVAL_SECONDS` (86400) | trigger 모드 실행 주기(초) |
| `--duration` | 없음 | trigger 모드에서 이 시간(초) 뒤 자동 종료 |
| `--keyword` / `--days` / `--max-results` | `.env` 값 | 파이프라인 파라미터 (두 모드 공통) |

모드별 종료 동작이 다르다. `single`은 파이프라인이 실패하면 종료 코드 1을 반환하므로 cron/CI가
실패를 감지할 수 있다. `trigger`는 한 주기의 실패를 로그와 실패 카운트로만 남기고 다음 주기를
계속 돌며, 종료 시 성공/실패 횟수를 요약 출력한다.

실행 전에 DB 컨테이너가 떠 있어야 한다(위 "DB" 섹션). 첫 실행 시 테이블이 없으면
`init_db()`가 생성한다.

## 파이프라인 흐름 (고정 순서, 분기 없음)

`fetch_news` → `store_raw` → `summarize` → `extract_insight` → `format_email` → `send_email`

각 노드는 `graph/pipeline.py`의 `PipelineState`(TypedDict)를 입출력으로 공유한다. 노드 하나가 실패하면 `error` 필드에 기록되고 `main.py`가 비정상 종료 코드를 반환한다.

## 스케줄링

`.github/workflows/daily-digest.yml`은 매일 08:00 KST에 `main.py`를 실행하는 GitHub Actions 예시다. 리포지토리 Secrets에 API 키를 등록하면 그대로 동작한다. 서버에서 직접 돌린다면 동일한 명령을 cron에 등록하면 된다.

```
0 8 * * * cd /path/to/news-insight-agent && /usr/bin/python3 main.py >> logs/run.log 2>&1
```

## 트리거 (프로세스 내 주기 실행)

cron/GitHub Actions 없이 프로세스 안에서 주기 실행이 필요할 때 쓰는 트리거 계층이다
(설계문서 3.1). 표준 라이브러리만 사용하며 `main.py --mode trigger`가 이 스케줄러를 쓴다.

트리거 자체를 직접 돌릴 수도 있다. `trigger/jobs.py`의 `JOBS`에 등록된 작업을 이름으로
지정한다 (`hello`=동작 확인용 hello_world, `digest`=`main.run_digest` 호출).

```bash
python -m trigger                          # hello를 10초마다 실행 (Ctrl+C 종료)
python -m trigger --interval 5             # 5초 주기
python -m trigger --duration 25            # 25초만 돌고 종료
python -m trigger --once                   # 1회만 실행
python -m trigger --job digest --interval 3600   # 파이프라인을 1시간 주기로
```

특성:

- **고정 주기**: 작업 소요 시간과 무관하게 `이전 예정 시각 + interval`로 다음 실행을 잡는다.
  10초 주기면 작업이 1초 걸려도 10초마다 실행된다. 한 주기를 넘길 만큼 늦어지면 밀린 실행은
  몰아서 따라잡지 않고 건너뛴다.
- **단일 스레드 순차 실행**: 긴 작업은 다른 작업을 지연시키므로 무거운 파이프라인은 주기를
  넉넉히 잡는다.
- **작업 예외 격리**: 작업이 던진 예외는 로그로 남기고 `Job.error_count`에 집계하며, 트리거는
  다음 주기를 계속 돈다.

새 작업을 붙이려면 `trigger/jobs.py`에 함수를 추가하고 `JOBS`에 등록하면 `--job <이름>`으로
바로 쓸 수 있다. 코드에서 직접 쓸 때는 다음과 같다.

```python
from trigger import Scheduler
from trigger.jobs import hello_world

scheduler = Scheduler()
scheduler.add_job(hello_world, interval=10.0)
scheduler.run_forever()   # 또는 start() / stop()
```

## 테스트

```bash
python -m unittest discover -t .                           # 전체 (tests/ + rag/tests/)
python -m unittest tests.test_scheduler                    # 스케줄러
python -m unittest tests.test_main_modes                   # main의 single/trigger 모드
python -m unittest tests.test_db_connection                # DB 접속 (URL 조립 + 실접속)
python -m unittest tests.test_db_crud                      # 테이블별 CRUD·제약·트랜잭션
python -m unittest tests.test_db                           # db.py 헬퍼 함수
RUN_SLOW_TESTS=1 python -m unittest tests.test_scheduler   # 실제 10초 주기 검증 포함 (약 21초)
```

빠른 테스트는 가짜 시계를 주입해 실제 대기 없이 10초 주기·지연 시 건너뛰기·예외 격리·다중
작업 주기를 검증한다. `test_main_modes`는 `run_pipeline`/`init_db`를 mock으로 대체해 외부
API 호출이나 이메일 발송 없이 모드별 동작(종료 코드, 주기 반복, 실패 후 계속 실행)을 확인한다.
실제 시간으로 10초 간격을 재는 통합 테스트는 시간이 걸리므로 `RUN_SLOW_TESTS=1`일 때만 실행된다.
`test_db*`는 실제 PostgreSQL에 붙는 통합 테스트로, DB에 접속할 수 없으면 skip된다. 테스트가
만든 행은 tearDown에서 삭제한다(공용 헬퍼: `tests/dbhelpers.py`).

## 다음 확장 (백로그, `mvp-implementation-breakdown.md` 2단계 참고)

Discord 발송(#9), 검증/품질게이트(#10), 기술문서 수집(#11), 중복제거/클러스터링(#12), 실시간 트리거(#13)는 아직 구현하지 않았다. `tools/`에 새 도구를 추가하고 `graph/pipeline.py`에 노드/엣지를 붙이는 방식으로 확장하면 된다. 예를 들어 Discord는 `tools/discord_send.py`를 만들고 `format_email` 다음에 `send_discord` 노드를 병렬로 붙이면 된다. 검증 게이트를 추가할 때 비로소 `add_conditional_edges`로 "재작업 여부"를 LLM이 판단하게 해, 고정 파이프라인에서 에이전틱 파이프라인으로 전환할 수 있다.

## 참고 문서

- `docs/database.md`: DB 접속 정보·테이블 구조·쿼리 모음
- `docs/docker.md`: 컨테이너 빌드/실행/영속화

- GNews Search Endpoint: https://docs.gnews.io/endpoints/search-endpoint
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- Resend API: https://resend.com/docs/api-reference/emails/send-email
