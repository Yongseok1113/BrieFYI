# prepare_data.py -> train.py -> evaluate.py를 순서대로 실행 (design doc 6.1절).
#
# 사용법:
#   scripts/run_pipeline.ps1 configs/qlora_qwen3-8b.yaml
#   scripts/run_pipeline.ps1 configs/smoke.yaml -SkipPrepare
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [switch]$SkipPrepare
)
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not $SkipPrepare) {
    python scripts/prepare_data.py --sources digest_pipeline
}

python -m summarize_ft.train --config $Config

$OutputDir = python -c "from summarize_ft.config import load_config; print(load_config('$Config').output_dir)"
$Testset = "data/golden/testset_v1.jsonl"

if (Test-Path $Testset) {
    python -m summarize_ft.evaluate --checkpoint $OutputDir --testset $Testset
} else {
    Write-Host "[skip] $Testset 없음 — evaluate는 골든 테스트셋 준비 후 수동 실행하세요."
}
