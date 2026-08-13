FROM python:3.11-slim

WORKDIR /app

# requirements만 먼저 복사해 의존성 레이어를 캐싱한다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# SQLite 데이터는 이미지가 아니라 볼륨(docker-compose.yml의 ./data)에 영속화한다.
RUN mkdir -p /app/data
ENV DB_PATH=/app/data/pipeline.db

# main.py는 실행 후 종료되는 배치 잡이다(상시 서버 아님).
ENTRYPOINT ["python", "main.py"]
