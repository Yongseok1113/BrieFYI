# Docker 실행 가이드

도커로 감싸는 대상은 두 개다. 파이프라인 실행 앱(`digest` = `main.py` + 의존성)과 저장소(`db` = PostgreSQL 16). GNews/Anthropic/Resend는 외부 API라 컨테이너화 대상이 아니고, LangGraph는 앱 프로세스 안에서 쓰는 라이브러리라서 별도 서비스가 필요 없다.

| 서비스 | 이미지 | 성격 |
| --- | --- | --- |
| `db` | `Dockerfile.db` (postgres:16-alpine + `db/schema.sql`) | 상시 실행. 데이터는 `pgdata` 볼륨 |
| `digest` | `Dockerfile` (python:3.11-slim) | 배치 잡. single 모드는 실행 후 종료 |

## 빌드

```bash
docker compose build
```

## DB만 띄우기

앱은 호스트에서 `python main.py`로 돌리고 DB만 컨테이너로 쓰는 개발 구성이 가장 편하다.

```powershell
.\scripts\db-up.ps1          # Windows
```

```bash
./scripts/db-up.sh           # macOS / Linux
```

스크립트는 이미지 빌드 → `db` 기동 → `pg_isready`로 준비 대기 → 접속 정보 출력까지 한다.
`-Psql`/`--psql`로 psql 세션, `-Down`/`--down`으로 정지(데이터 유지),
`-Reset -Force`/`--reset --force`로 볼륨까지 삭제 후 재생성한다.

스키마는 컨테이너 데이터 디렉터리가 비어 있을 때만 `/docker-entrypoint-initdb.d`에서
자동 적용된다. 즉 이미 만들어진 볼륨에는 다시 적용되지 않으므로, 스키마를 바꿨다면
앱의 `init_db()`(`CREATE TABLE IF NOT EXISTS`)에 의존하거나 볼륨을 리셋한다.

## 1회 실행 (수동 테스트)

```bash
docker compose run --rm digest
# 파라미터를 바꾸려면
docker compose run --rm digest --keyword "AI" --days 1 --max-results 10
# 컨테이너 안에서 주기 실행(trigger 모드)
docker compose run --rm digest --mode trigger --interval 3600
```

`digest`는 `depends_on: db (service_healthy)`라서 DB가 준비된 뒤에 시작한다. single 모드의 `main.py`는 실행 후 종료되는 배치 잡이므로 `docker compose up`으로 상시 띄워둘 필요는 없다.

## 데이터 영속화

DB 데이터는 이미지가 아니라 `pgdata` 네임드 볼륨(`docker-compose.yml`의 `volumes`)에 저장된다. `db` 컨테이너를 지우고 다시 만들어도 데이터는 유지되며, `docker compose down -v` 또는 `db-up` 스크립트의 리셋 옵션을 쓸 때만 삭제된다.

앱은 `DATABASE_URL`로만 DB를 찾는다. 컨테이너 내부에서는 호스트가 `db`(서비스명), 호스트에서 직접 실행할 때는 `localhost:5432`다. 관리형 DB(RDS/Supabase 등)로 옮길 때는 `DATABASE_URL`만 바꾸고 `db` 서비스를 제거하면 된다.

## 시크릿

API 키는 이미지에 절대 포함하지 않는다. `.env` 파일을 프로젝트 루트에 두면 `docker-compose.yml`의 `env_file`이 런타임에 주입한다. `.env`는 `.dockerignore`와 `.gitignore`에 모두 포함되어야 한다(레포에 `.gitignore`가 없다면 추가 필요). `docker compose config`는 `.env` 값을 평문으로 출력하니 그 결과를 붙여넣지 않도록 주의한다.

DB 비밀번호(`POSTGRES_PASSWORD`)도 `.env`에서 읽는다. 기본값은 로컬 개발용 `briefyi`이므로, 컨테이너 포트를 외부에 노출하는 환경에서는 반드시 바꾼다.

## 스케줄링

컨테이너 안에 cron을 넣기보다, 호스트 cron이 컨테이너를 주기적으로 실행하는 방식을 권장한다. 컨테이너를 상시 켜둘 필요가 없고 구성이 단순하다.

```
0 8 * * * cd /path/to/BrieFYI && docker compose run --rm digest >> logs/run.log 2>&1
```

GitHub Actions(`.github/workflows/daily-digest.yml`)로 이미 스케줄링 중이라면 Docker는 로컬 개발/수동 실행/자체 서버 배포용으로만 쓰고, CI에서는 굳이 이미지를 빌드하지 않고 `pip install` 방식을 유지해도 된다. 두 방식을 동시에 쓸 필요는 없다.

## 향후 확장 시 고려사항

Discord 봇처럼 상시 연결이 필요한 기능을 추가하면, 그건 배치 잡이 아니라 별도의 상시 실행 서비스 컨테이너로 분리해야 한다. 벡터 검색(#12 중복제거/클러스터링)이 필요해지면 별도 컨테이너를 띄우기보다 `Dockerfile.db`를 `pgvector/pgvector:pg16` 기반으로 바꾸고 `CREATE EXTENSION vector`를 초기화 스크립트에 추가하는 편이 구성이 단순하다.
