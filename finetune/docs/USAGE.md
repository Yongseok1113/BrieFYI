# summarize_ft 사용법

## 0. 설치

```bash
cd finetune
bash scripts/setup.sh          # Windows는 scripts/setup.ps1
python scripts/check_env.py    # GPU/CUDA/패키지 버전 점검
```

## 1. 스모크 테스트 (가장 먼저 할 것)

GPU가 없거나 실제 데이터/API 키가 아직 없어도 파이프라인이 끝까지 도는지 몇 분 안에 확인한다.

```bash
python -m summarize_ft.train --config configs/smoke.yaml
```

`data/sample/`의 8건/3건짜리 번들 샘플로 돌아간다. 여기서 에러가 나면 실제 학습 전에 반드시 고쳐야 한다.

## 2. 학습 데이터 준비

```bash
# 자체 파이프라인(digests 테이블)만 사용
python scripts/prepare_data.py --sources digest_pipeline

# AI Hub + Dacon까지 함께 사용 (다중 소스)
python scripts/prepare_data.py \
    --sources digest_pipeline aihub dacon \
    --aihub-dir /path/to/aihub_raw \
    --dacon-csv /path/to/dacon/train.csv
```

`data/processed/summarize_train.jsonl`, `summarize_val.jsonl`이 생성된다. `digests_export.py`는 repo 루트의
`db/db.py`(PostgreSQL, `config.DATABASE_URL`)에 직접 접속하므로 `.env`가 repo 루트에 설정돼 있어야 한다.

## 3. 학습

```bash
python -m summarize_ft.train --config configs/qlora_qwen3-8b.yaml
python -m summarize_ft.train --config configs/qlora_gemma4-12b.yaml
python -m summarize_ft.train --config configs/qlora_qwen3-8b.yaml --set train.epochs=1 lora.r=32
```

코랩에서 세션이 끊기면 같은 명령을 다시 실행하면 된다 — `train.py`가 `output_dir` 안의 마지막
`checkpoint-*`를 찾아 자동으로 이어서 학습한다 (`resume_from_checkpoint`).

## 4. 평가 (4단계)

```bash
python -m summarize_ft.evaluate \
    --checkpoint runs/qwen3-8b-summarize-v1 \
    --testset data/golden/testset_v1.jsonl \
    --with-grounding   # 3단계 grounding check까지 (Claude API 호출, ANTHROPIC_API_KEY 필요)
```

리포트는 `<checkpoint>/eval_report.json`에 저장된다.

## 5. 어댑터 병합 (선택)

vLLM/TGI 멀티 adapter 서빙이 아니라 단일 체크포인트로 배포하고 싶을 때만 사용한다.

```bash
python -m summarize_ft.merge_lora \
    --base Qwen/Qwen3-8B-Instruct \
    --adapter runs/qwen3-8b-summarize-v1 \
    --out runs/qwen3-8b-summarize-v1-merged
```

## 6. 전체 파이프라인 한 번에

```bash
scripts/run_pipeline.sh configs/qlora_qwen3-8b.yaml
```

## 7. 테스트

```bash
pytest finetune/tests
```

모델 다운로드가 전혀 필요 없다 — `schema.py`/`config.py`/`data.py` 등 로직만 검증한다.
