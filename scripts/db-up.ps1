<#
.SYNOPSIS
    BrieFYI PostgreSQL 컨테이너(docker-compose의 db 서비스)를 띄우고 준비될 때까지 기다린다.

.DESCRIPTION
    기본 동작: 이미지 빌드 -> db 컨테이너 기동 -> pg_isready로 접속 가능해질 때까지 대기 ->
    접속 정보 출력. 이미 떠 있으면 그대로 두고 상태만 확인한다.

.PARAMETER Down
    db 컨테이너를 정지한다. 데이터(pgdata 볼륨)는 유지된다.

.PARAMETER Reset
    데이터 볼륨까지 삭제하고 스키마부터 다시 만든다. 저장된 기사/다이제스트가 모두 사라지므로
    -Force와 함께 써야 한다.

.PARAMETER Logs
    기동 후 컨테이너 로그를 따라 출력한다(Ctrl+C로 중단).

.PARAMETER Psql
    기동 후 컨테이너 안의 psql 세션을 연다.

.EXAMPLE
    .\scripts\db-up.ps1
    .\scripts\db-up.ps1 -Psql
    .\scripts\db-up.ps1 -Reset -Force
    .\scripts\db-up.ps1 -Down
#>
[CmdletBinding()]
param(
    [switch]$Down,
    [switch]$Reset,
    [switch]$Force,
    [switch]$Logs,
    [switch]$Psql,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Fail {
    param([string]$Message)

    Write-Host $Message -ForegroundColor Red
    Pop-Location -ErrorAction SilentlyContinue
    exit 1
}

function Test-DockerEngine {
    # 네이티브 명령의 stderr 리다이렉트가 종료 예외로 승격되지 않게 잠시 완화한다.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $null = docker info 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Get-Setting {
    param([string]$Name, [string]$Default)

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }

    $envFile = Join-Path $root '.env'
    if (Test-Path $envFile) {
        $match = Select-String -Path $envFile -Pattern "^\s*$Name\s*=" | Select-Object -First 1
        if ($match) {
            $parsed = ($match.Line -split '=', 2)[1].Trim().Trim('"').Trim("'")
            if ($parsed) { return $parsed }
        }
    }
    return $Default
}

Push-Location $root
try {
    # 도커 엔진 확인 (Docker Desktop이 꺼져 있으면 여기서 멈춘다)
    if (-not (Test-DockerEngine)) {
        Fail "도커 엔진에 연결할 수 없다. Docker Desktop을 실행한 뒤 다시 시도할 것."
    }

    $dbName = Get-Setting -Name 'POSTGRES_DB' -Default 'briefyi'
    $dbUser = Get-Setting -Name 'POSTGRES_USER' -Default 'briefyi'
    $dbPort = Get-Setting -Name 'POSTGRES_PORT' -Default '5432'

    if ($Down) {
        Write-Host "db 컨테이너를 정지한다 (데이터는 유지)..."
        docker compose stop db
        if ($LASTEXITCODE -ne 0) { Fail "docker compose stop 실패" }
        Write-Host "정지 완료. 데이터 볼륨(pgdata)은 그대로 남아 있다."
        return
    }

    if ($Reset) {
        if (-not $Force) {
            Fail "-Reset은 pgdata 볼륨을 삭제해 저장된 데이터를 모두 지운다. 확인했다면 -Force를 함께 지정할 것."
        }
        Write-Host "컨테이너와 데이터 볼륨을 삭제한다..."
        docker compose down -v
        if ($LASTEXITCODE -ne 0) { Fail "docker compose down -v 실패" }
    }

    Write-Host "db 이미지 빌드 및 기동..."
    docker compose up -d --build db
    if ($LASTEXITCODE -ne 0) { Fail "docker compose up 실패" }

    Write-Host "PostgreSQL 준비 대기 (최대 $TimeoutSeconds 초)..." -NoNewline
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        $null = docker compose exec -T db pg_isready -U $dbUser -d $dbName
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    Write-Host ""

    if (-not $ready) {
        docker compose logs --tail 40 db
        Fail "PostgreSQL이 ${TimeoutSeconds}초 안에 준비되지 않았다. 위 로그를 확인할 것."
    }

    Write-Host "준비 완료." -ForegroundColor Green
    docker compose ps db
    Write-Host ""
    Write-Host "호스트에서 접속:  postgresql://${dbUser}:***@localhost:${dbPort}/${dbName}"
    Write-Host "테이블 확인:      docker compose exec db psql -U $dbUser -d $dbName -c '\dt'"
    Write-Host "파이프라인 실행:  python main.py --mode single"

    if ($Psql) {
        docker compose exec db psql -U $dbUser -d $dbName
    }
    elseif ($Logs) {
        docker compose logs -f db
    }
}
finally {
    Pop-Location
}
