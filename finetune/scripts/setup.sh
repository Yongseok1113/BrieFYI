#!/usr/bin/env bash
# finetune/ 의존성 설치 (WSL/Linux/코랩용).
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

echo "설치 완료. 다음으로 scripts/check_env.py를 돌려 환경을 점검하세요."
