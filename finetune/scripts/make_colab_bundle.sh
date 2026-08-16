#!/usr/bin/env bash
# 코랩에 업로드할 finetune_bundle.zip을 만든다.
# 학습에 필요한 코드/설정만 담고, 무거운 data/processed, runs/는 제외한다
# (학습 데이터는 로컬에서 scripts/prepare_data.py로 따로 뽑아 별도 업로드).
set -euo pipefail

cd "$(dirname "$0")/.."   # finetune/ 로 이동

OUT="finetune_bundle.zip"
rm -f "$OUT"

zip -r "$OUT" \
    src \
    configs \
    scripts/check_env.py \
    requirements-colab.txt \
    pyproject.toml \
    -x "*.pyc" -x "__pycache__/*"

echo "생성 완료: finetune/$OUT"
echo "이 파일을 코랩 노트북의 업로드 셀에서 선택하세요."
