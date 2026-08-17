FROM python:3.11-slim

WORKDIR /app

# 필요한 내용만 이 폴더 자체에서 수집해서 이미지로 빌드한다 — 레포 루트 코드를
# 참조하지 않는 자체 완결 이미지다 (docs/data-pipeline-design.md 8절).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "data_pipeline.cli"]
CMD ["run", "--stage", "all"]
