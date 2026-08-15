# DB 가이드 (PostgreSQL 16)

접속 정보와 테이블 구조를 한곳에 모은 문서다. 컨테이너 기동/운영은 `docker.md`, 코드 접근은
`db/db.py`를 참고한다.

## 1. 컨테이너 정보

| 항목 | 값 |
| --- | --- |
| compose 서비스명 | `db` |
| 컨테이너명 | `briefyi-db-1` |
| 이미지 | `briefyi-db:16` (`Dockerfile.db` = `postgres:16-alpine` + `db/schema.sql`) |
| 서버 버전 | PostgreSQL 16.14 (alpine, musl) |
| 데이터 볼륨 | `briefyi_pgdata` → 컨테이너 내부 `/var/lib/postgresql/data` |
| 포트 매핑 | 호스트 `${POSTGRES_PORT:-5432}` → 컨테이너 `5432` |
| 서버 인코딩 | `UTF8` (`POSTGRES_INITDB_ARGS=--encoding=UTF8`) |
| 기동 | `.\scripts\db-up.ps1` / `./scripts/db-up.sh` |

## 2. 계정·접속 정보

기본값은 로컬 개발용이다. 포트를 외부에 노출하는 환경에서는 `.env`에서 비밀번호를 반드시 바꾼다.

| 항목 | 환경변수 | 기본값 |
| --- | --- | --- |
| DB 이름 | `POSTGRES_DB` | `briefyi` |
| 사용자 | `POSTGRES_USER` | `briefyi` (superuser, 컨테이너 초기화 시 생성) |
| 비밀번호 | `POSTGRES_PASSWORD` | `briefyi` |
| 호스트 | `POSTGRES_HOST` | `localhost` (컨테이너 내부에서는 `db`) |
| 포트 | `POSTGRES_PORT` | `5432` |
| 접속 문자열 | `DATABASE_URL` | 위 값들로 조립 |

`.env`의 같은 변수들이 두 곳에 동시에 쓰인다. `docker-compose.yml`이 컨테이너 초기화
(`POSTGRES_*`)에 쓰고, `config.py`가 앱 접속 정보에 쓴다. 그래서 한 곳만 바꿔도 양쪽이 맞는다.

### 접속 문자열 결정 순서 (`config.py`)

1. `DATABASE_URL` 환경변수가 있으면 **그 값을 그대로** 쓴다 (관리형 DB로 옮길 때 이것만 주면 된다).
2. 없으면 `postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}`로 조립한다. 사용자명·비밀번호는
   URL 인코딩(`quote_plus`)하므로 `@`, `:`, `/` 같은 문자가 들어가도 안전하다.

### 실행 위치별 접속 대상

| 실행 위치 | 호스트 | 설정 주체 |
| --- | --- | --- |
| 호스트에서 `python main.py` | `localhost:5432` | `config.py` 기본값 (또는 `.env`) |
| `docker compose run digest` | `db:5432` | `docker-compose.yml`의 `environment.DATABASE_URL` |
| GitHub Actions | `localhost:5432` (서비스 컨테이너) | 워크플로 `env.DATABASE_URL` |

### psql / 외부 클라이언트

```bash
# 컨테이너 안에서 (비밀번호 불필요: 로컬 소켓은 trust 인증)
docker compose exec db psql -U briefyi -d briefyi

# 호스트에서 psql이 설치돼 있다면
psql "postgresql://briefyi:briefyi@localhost:5432/briefyi"
```

DBeaver·pgAdmin 등 GUI 클라이언트는 Host `localhost`, Port `5432`, Database `briefyi`,
User `briefyi`, Password `briefyi`(또는 `.env`에 설정한 값)로 접속한다.

## 3. 테이블

스키마 정의는 `db/schema.sql` 하나이며, 컨테이너 최초 기동 시
`/docker-entrypoint-initdb.d/01-schema.sql`로 자동 적용되고 앱 시작 시 `init_db()`가
`CREATE TABLE IF NOT EXISTS`로 한 번 더 확인한다.

```
raw_articles (수집 원문)        digests (요약·인사이트) 1 ──< send_log (발송 이력)
        │                                                        ON DELETE CASCADE
        └─ 직접적인 FK 관계는 없다. digest_date/keyword로 논리적으로 연결된다.
```

### `raw_articles` — 수집된 기사 원문 (구현 항목 #2)

| 컬럼 | 타입 | NULL | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `id` | `bigint` (BIGSERIAL) | NO | 시퀀스 | PK |
| `digest_date` | `date` | NO | | 이 기사를 수집한 다이제스트 기준일 |
| `title` | `text` | NO | | 기사 제목 |
| `description` | `text` | YES | | GNews 요약문 |
| `url` | `text` | NO | | 원문 URL, **UNIQUE** (중복 수집 방지 키) |
| `source` | `text` | YES | | 매체명 |
| `published_at` | `timestamptz` | YES | | 발행 시각. 값이 없으면 NULL |
| `fetched_at` | `timestamptz` | NO | `now()` | 저장 시각 |

제약·인덱스: `raw_articles_pkey(id)`, `raw_articles_url_key(url)` UNIQUE,
`raw_articles_digest_date_idx(digest_date)`.

### `digests` — 요약/인사이트/시사점 결과 (구현 항목 #3, #4)

| 컬럼 | 타입 | NULL | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `id` | `bigint` (BIGSERIAL) | NO | 시퀀스 | PK. `send_log`가 참조 |
| `digest_date` | `date` | NO | | 다이제스트 기준일 |
| `keyword` | `text` | NO | | 실행 시 사용한 검색 키워드 |
| `summary_json` | `jsonb` | NO | | 기사별 요약 배열 |
| `insight_json` | `jsonb` | NO | | 인사이트/비즈니스 시사점 객체 |
| `created_at` | `timestamptz` | NO | `now()` | 생성 시각 |

제약·인덱스: `digests_pkey(id)`, `digests_digest_date_idx(digest_date)`.

`jsonb`이므로 파이썬 dict/list를 그대로 넣고 그대로 받는다(`psycopg`의 `Jsonb` 어댑터).
`insight_json->>'implication'` 처럼 SQL에서 직접 조회할 수도 있다.

### `send_log` — 채널별 발송 이력 (구현 항목 #6)

| 컬럼 | 타입 | NULL | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `id` | `bigint` (BIGSERIAL) | NO | 시퀀스 | PK |
| `digest_id` | `bigint` | NO | | `digests(id)` FK, **ON DELETE CASCADE** |
| `channel` | `text` | NO | | `email` 등 채널 이름 |
| `recipient` | `text` | YES | | 수신자 |
| `status` | `text` | NO | | `success` / `failed` |
| `error` | `text` | YES | | 실패 사유 |
| `sent_at` | `timestamptz` | NO | `now()` | 발송 시각 |

제약·인덱스: `send_log_pkey(id)`, `send_log_digest_id_fkey → digests(id) ON DELETE CASCADE`,
`send_log_digest_id_idx(digest_id)`.

## 4. 코드에서의 접근

모든 DB 접근은 `db/db.py`만 통과한다(psycopg 3, `row_factory=dict_row`).

| 함수 | 동작 |
| --- | --- |
| `init_db()` | `schema.sql` 적용 (idempotent) |
| `get_conn()` | 커넥션 컨텍스트 매니저. 정상 종료 시 커밋, 예외 시 롤백 |
| `insert_articles(digest_date, articles) -> int` | `ON CONFLICT (url) DO NOTHING`으로 저장, 신규 건수 반환 |
| `save_digest(digest_date, keyword, summary, insight) -> int` | `RETURNING id`로 새 digest id 반환 |
| `log_send(digest_id, channel, recipient, status, error=None)` | 발송 이력 1건 기록 |

```python
from db.db import get_conn

with get_conn() as conn:
    rows = conn.execute(
        "SELECT id, title FROM raw_articles WHERE digest_date = %s ORDER BY id", ("2026-08-14",)
    ).fetchall()   # dict_row -> [{'id': 1, 'title': '...'}, ...]
```

## 5. 자주 쓰는 쿼리

```sql
-- 날짜별 수집 건수
SELECT digest_date, count(*) FROM raw_articles GROUP BY 1 ORDER BY 1 DESC;

-- 최근 다이제스트와 발송 결과
SELECT d.id, d.digest_date, d.keyword, s.channel, s.status, s.sent_at
FROM digests d LEFT JOIN send_log s ON s.digest_id = d.id
ORDER BY d.id DESC LIMIT 10;

-- 인사이트 본문만 꺼내기 (jsonb)
SELECT id, jsonb_array_length(summary_json) AS 요약수, insight_json->>'implication' AS 시사점
FROM digests ORDER BY id DESC LIMIT 5;

-- 발송 실패 목록
SELECT digest_id, recipient, error, sent_at FROM send_log WHERE status <> 'success' ORDER BY sent_at DESC;
```

## 6. 테스트

DB가 떠 있어야 하며, 접속할 수 없으면 자동 skip된다. 테스트가 만든 행은 tearDown에서 지운다
(`keyword='__test__'`, `url='https://test.invalid/...'`).

```bash
python -m unittest tests.test_db_connection   # 접속 (URL 조립, 실접속, 인코딩, 실패 케이스)
python -m unittest tests.test_db_crud         # 테이블별 CRUD, 제약, CASCADE, 트랜잭션
python -m unittest tests.test_db              # db.py 헬퍼 함수
```

## 7. 트러블슈팅

| 증상 | 원인/해결 |
| --- | --- |
| `connection refused` | 컨테이너가 안 떠 있다. `.\scripts\db-up.ps1` 실행 |
| `port is already allocated` | 호스트 5432가 이미 사용 중. `.env`에 `POSTGRES_PORT=5433` 지정 후 재기동 |
| `password authentication failed` | `.env`의 비밀번호를 바꿨는데 볼륨은 옛 비밀번호로 초기화된 상태. 계정 정보는 최초 기동 시에만 반영되므로 `db-up.ps1 -Reset -Force`로 볼륨을 다시 만들거나 `ALTER USER`로 변경 |
| 스키마를 바꿨는데 반영 안 됨 | 초기화 스크립트는 빈 볼륨에서만 실행된다. `IF NOT EXISTS`로 커버되지 않는 변경(컬럼 타입 등)은 `-Reset -Force` 또는 수동 `ALTER TABLE` |
| 한글이 깨짐 | 서버 인코딩은 UTF8이다. 클라이언트 쪽 `client_encoding`이나 터미널 코드페이지를 확인 |
