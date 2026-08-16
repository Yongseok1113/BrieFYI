# finetune/ 의존성 설치 (Windows PowerShell용).
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

Write-Host "설치 완료. 다음으로 scripts/check_env.py를 돌려 환경을 점검하세요."
