#!/usr/bin/env bash
# BrieFYI PostgreSQL 컨테이너(docker-compose의 db 서비스)를 띄우고 준비될 때까지 기다린다.
# db-up.ps1의 POSIX 버전.
#
# 사용법:
#   ./scripts/db-up.sh              # 빌드 + 기동 + 준비 대기
#   ./scripts/db-up.sh --psql       # 기동 후 psql 세션
#   ./scripts/db-up.sh --logs       # 기동 후 로그 팔로우
#   ./scripts/db-up.sh --down       # 정지 (데이터 유지)
#   ./scripts/db-up.sh --reset --force   # 데이터 볼륨까지 삭제하고 재생성
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOWN=0
RESET=0
FORCE=0
LOGS=0
PSQL=0
TIMEOUT=90

while [ $# -gt 0 ]; do
    case "$1" in
        --down) DOWN=1 ;;
        --reset) RESET=1 ;;
        --force) FORCE=1 ;;
        --logs) LOGS=1 ;;
        --psql) PSQL=1 ;;
        --timeout) TIMEOUT="$2"; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
    esac
    shift
done

# .env 값 우선순위: 실제 환경변수 > .env 파일 > 기본값
setting() {
    local name="$1" default="$2" value
    value="${!name:-}"
    if [ -n "$value" ]; then
        echo "$value"
        return
    fi
    if [ -f .env ]; then
        value="$(sed -n "s/^[[:space:]]*${name}[[:space:]]*=[[:space:]]*//p" .env | head -1 | tr -d '"'"'" | tr -d '\r')"
        if [ -n "$value" ]; then
            echo "$value"
            return
        fi
    fi
    echo "$default"
}

if ! docker info >/dev/null 2>&1; then
    echo "도커 엔진에 연결할 수 없다. 도커가 실행 중인지 확인할 것." >&2
    exit 1
fi

DB_NAME="$(setting POSTGRES_DB briefyi)"
DB_USER="$(setting POSTGRES_USER briefyi)"
DB_PORT="$(setting POSTGRES_PORT 5432)"

if [ "$DOWN" -eq 1 ]; then
    echo "db 컨테이너를 정지한다 (데이터는 유지)..."
    docker compose stop db
    echo "정지 완료. 데이터 볼륨(pgdata)은 그대로 남아 있다."
    exit 0
fi

if [ "$RESET" -eq 1 ]; then
    if [ "$FORCE" -ne 1 ]; then
        echo "--reset은 pgdata 볼륨을 삭제해 저장된 데이터를 모두 지운다. 확인했다면 --force를 함께 지정할 것." >&2
        exit 2
    fi
    echo "컨테이너와 데이터 볼륨을 삭제한다..."
    docker compose down -v
fi

echo "db 이미지 빌드 및 기동..."
docker compose up -d --build db

printf "PostgreSQL 준비 대기 (최대 %s초)..." "$TIMEOUT"
deadline=$(( $(date +%s) + TIMEOUT ))
ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if docker compose exec -T db pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        ready=1
        break
    fi
    printf "."
    sleep 2
done
printf "\n"

if [ "$ready" -ne 1 ]; then
    docker compose logs --tail 40 db
    echo "PostgreSQL이 ${TIMEOUT}초 안에 준비되지 않았다. 위 로그를 확인할 것." >&2
    exit 1
fi

echo "준비 완료."
docker compose ps db
echo
echo "호스트에서 접속:  postgresql://${DB_USER}:***@localhost:${DB_PORT}/${DB_NAME}"
echo "테이블 확인:      docker compose exec db psql -U ${DB_USER} -d ${DB_NAME} -c '\\dt'"
echo "파이프라인 실행:  python main.py --mode single"

if [ "$PSQL" -eq 1 ]; then
    docker compose exec db psql -U "$DB_USER" -d "$DB_NAME"
elif [ "$LOGS" -eq 1 ]; then
    docker compose logs -f db
fi
