# 코랩에 업로드할 finetune_bundle.zip을 만든다 (Windows PowerShell용).
# 학습에 필요한 코드/설정만 담고, 무거운 data/processed, runs/는 제외한다.
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")   # finetune/ 로 이동

$Out = "finetune_bundle.zip"
if (Test-Path $Out) { Remove-Item $Out }

$Items = @("src", "configs", "scripts\check_env.py", "requirements-colab.txt", "pyproject.toml")
Compress-Archive -Path $Items -DestinationPath $Out

Write-Host "생성 완료: finetune\$Out"
Write-Host "이 파일을 코랩 노트북의 업로드 셀에서 선택하세요."
