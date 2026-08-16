#!/usr/bin/env bash
# prepare_data.py -> train.py -> evaluate.py를 순서대로 실행 (design doc 6.1절).
#
# 사용법:
#   scripts/run_pipeline.sh configs/qlora_qwen3-8b.yaml
#   scripts/run_pipeline.sh configs/smoke.yaml --skip-prepare   # 샘플 데이터로 바로 학습만
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:?사용법: run_pipeline.sh <config.yaml> [--skip-prepare]}"
shift || true

if [[ "${1:-}" != "--skip-prepare" ]]; then
    python3 scripts/prepare_data.py --sources digest_pipeline
fi

python3 -m summarize_ft.train --config "$CONFIG"

OUTPUT_DIR=$(python3 -c "from summarize_ft.config import load_config; print(load_config('$CONFIG').output_dir)")
TESTSET="data/golden/testset_v1.jsonl"

if [[ -f "$TESTSET" ]]; then
    python3 -m summarize_ft.evaluate --checkpoint "$OUTPUT_DIR" --testset "$TESTSET"
else
    echo "[skip] $TESTSET 없음 — evaluate는 골든 테스트셋 준비 후 수동 실행하세요."
fi
