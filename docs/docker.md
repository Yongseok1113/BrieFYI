# Docker 실행 가이드

이 프로젝트에서 도커로 감싸는 대상은 파이프라인 실행 앱(`main.py` + 의존성) 하나뿐이다. GNews/Anthropic/Resend는 외부 API라 컨테이너화 대상이 아니고, LangGraph는 앱 프로세스 안에서 쓰는 라이브러리라서 별도 서비스가 필요 없다.

## 빌드

```bash
docker compose build
```

## 1회 실행 (수동 테스트)

```bash
docker compose run --rm digest
# 파라미터를 바꾸려면
docker compose run --rm digest --keyword "AI" --days 1 --max-results 10
```

`main.py`는 실행 후 종료되는 배치 잡이므로 `docker compose up`으로 상시 띄워둘 필요는 없다.

## 데이터 영속화

SQLite 파일은 이미지 안이 아니라 호스트의 `./data` 디렉터리에 볼륨으로 저장된다(`docker-compose.yml`의 `volumes`). 컨테이너를 지우고 다시 만들어도 `./data/pipeline.db`는 유지된다. `./data`는 로컬(네이티브) 파일시스템 경로여야 한다. 네트워크로 마운트된 드라이브 위에서는 SQLite의 파일 잠금이 깨질 수 있다.

## 시크릿

API 키는 이미지에 절대 포함하지 않는다. `.env` 파일(`.env.example` 참고)을 프로젝트 루트에 두면 `docker-compose.yml`의 `env_file`이 런타임에 주입한다. `.env`는 `.dockerignore`와 `.gitignore`에 모두 포함되어야 한다(레포에 `.gitignore`가 없다면 추가 필요).

## 스케줄링

컨테이너 안에 cron을 넣기보다, 호스트 cron이 컨테이너를 주기적으로 실행하는 방식을 권장한다. 컨테이너를 상시 켜둘 필요가 없고 구성이 단순하다.

```
0 8 * * * cd /path/to/BrieFYI && docker compose run --rm digest >> logs/run.log 2>&1
```

GitHub Actions(`.github/workflows/daily-digest.yml`)로 이미 스케줄링 중이라면 Docker는 로컬 개발/수동 실행/자체 서버 배포용으로만 쓰고, CI에서는 굳이 이미지를 빌드하지 않고 `pip install` 방식을 유지해도 된다. 두 방식을 동시에 쓸 필요는 없다.

## 향후 확장 시 고려사항

Discord 봇처럼 상시 연결이 필요한 기능을 추가하면, 그건 배치 잡이 아니라 별도의 상시 실행 서비스 컨테이너로 분리해야 한다. 벡터 DB(Chroma 등)를 도입하면 그 역시 별도 컨테이너 + 볼륨으로 추가한다.
