# data_pipeline

프롬프트 엔지니어링 기반 학습 데이터 생성 파이프라인. 설계 문서: `docs/data-pipeline-design.md`(레포 루트).

파인튜닝하지 않은 기본 오픈모델(기본값 Groq `llama-3.3-70b-versatile`, 카드 등록 없는 무료 티어)을 사용해
뉴스 기사에서 insights/implications/category/domain/entity/event를 뽑아내고, 통합 단어 테이블로
정규화한 뒤 `finetune/`의 학습 데이터로 export한다.

## 단계

```
ingest (수집, 구조화 소스는 필드 그대로 매핑) -> pipeline_status='pending'
extract (로컬 키워드 추출, API 호출 아님)      -> pipeline_status='extracted'
enrich (LLM 호출, 원시 메타데이터 생성)         -> pipeline_status='enriched'
normalize (fuzzy 우선 + LLM fallback 정규화)    -> pipeline_status='normalized'
```

## 로컬 실행

```bash
uv pip install -r data_pipeline/requirements.txt
uv pip install -e data_pipeline/

# .env는 레포 루트 하나로 통합 관리한다 (레포 루트 .env.example의 DATA_PIPELINE_* 항목 참고).
# data_pipeline/에는 별도 .env를 두지 않는다 — config.py가 루트 .env를 직접 읽는다.

python -m data_pipeline.cli run --stage all --limit 20
python -m data_pipeline.cli synonyms build
```

## Docker

```bash
cd ..
docker compose up -d db          # 메인 앱 Postgres 먼저 기동
cd data_pipeline
docker compose run --rm data_pipeline run --stage all --limit 20
docker compose run --rm data_pipeline synonyms build
```

## 테스트

```bash
pytest tests
```

DB/모델 다운로드 없이 순수 로직(rate_limiter, keywords 폴백, clustering, normalize의 exact/fuzzy 매칭)만 검증한다.

## finetune/ 연동

```bash
python finetune/scripts/prepare_data.py --sources enrichment
```
