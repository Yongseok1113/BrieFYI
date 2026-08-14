FROM python:3.11-slim

WORKDIR /app

# requirements만 먼저 복사해 의존성 레이어를 캐싱한다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 데이터는 별도의 db 서비스(PostgreSQL)에 저장한다. 접속 정보는 DATABASE_URL로 주입되며
# docker-compose.yml이 컨테이너 내부용 값(host=db)을 설정한다.

# main.py는 single 모드에서 실행 후 종료되는 배치 잡이다(상시 서버 아님).
ENTRYPOINT ["python", "main.py"]
