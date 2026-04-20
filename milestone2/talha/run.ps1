<#
.SYNOPSIS
    PowerShell launcher for Talha's M2 Comparative Benchmark.
    Talha Mudassar | PDC Milestone 2

.DESCRIPTION
    Manages building and running the comparative benchmark Docker container.
    Follows the same run.ps1 pattern used by all M1 teammates.

.PARAMETER Command
    benchmark  - Run full comparative benchmark (default)
    quick      - Run benchmark with 2 runs per config (faster)
    generate   - Only generate the test image
    shell      - Open interactive bash shell inside container

.PARAMETER Runs
    Number of timed runs per configuration (default: 3)

.PARAMETER Workers
    Space-separated worker counts to test (default: "1 2 4 8")

.PARAMETER SaveJson
    If specified, writes machine-readable JSON to ./results.json

.PARAMETER SkipShm
    Skip the shared-memory benchmark (useful if /dev/shm unavailable)

.PARAMETER SkipMproc
    Skip the multiprocessing benchmark

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Command quick
    .\run.ps1 -Runs 5 -SaveJson
    .\run.ps1 -Command shell
#>

param(
    [ValidateSet("benchmark", "quick", "generate", "shell")]
    [string]$Command   = "benchmark",
    [int]   $Runs      = 3,
    [string]$Workers   = "1 2 4 8",
    [switch]$SaveJson,
    [switch]$SkipShm,
    [switch]$SkipMproc
)

$ErrorActionPreference = "Stop"
$IMAGE_NAME = "m2-benchmark-talha"

# ── Resolve repo root (two levels up from this script's dir) ───────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = (Resolve-Path (Join-Path $ScriptDir ".." "..")).Path

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Milestone 2 – Comparative Benchmark" -ForegroundColor Cyan
Write-Host "  Talha Mudassar | PDC Spring 2025" -ForegroundColor Cyan
Write-Host "  Command  : $Command" -ForegroundColor Cyan
Write-Host "  Runs     : $Runs" -ForegroundColor Cyan
Write-Host "  Workers  : $Workers" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Build image ─────────────────────────────────────────────────────────────
Write-Host "[1/2] Building Docker image '$IMAGE_NAME'..." -ForegroundColor Yellow
# Build context is the repo root so COPY paths in Dockerfile resolve correctly
docker build `
    --tag "$IMAGE_NAME" `
    --file "$ScriptDir\Dockerfile" `
    "$RepoRoot"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Image built: $IMAGE_NAME" -ForegroundColor Green

# ── Run ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/2] Running container..." -ForegroundColor Yellow

$WorkerArgs  = $Workers -split " " | ForEach-Object { $_ }
$DockerFlags = @("run", "--rm")

# Mount output directory so results.json lands on the host
$OutputDir = Join-Path $ScriptDir "output"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
$DockerFlags += @("-v", "${OutputDir}:/output")

switch ($Command) {

    "benchmark" {
        $BenchArgs = @(
            "python", "/app/benchmark_comparative.py",
            "--image", "/app/test_image.jpg",
            "--runs",  "$Runs",
            "--workers"
        ) + $WorkerArgs

        if ($SkipShm)   { $BenchArgs += "--skip-shm" }
        if ($SkipMproc) { $BenchArgs += "--skip-multiprocessing" }
        if ($SaveJson)  { $BenchArgs += @("--output", "/output/results.json") }

        & docker @DockerFlags $IMAGE_NAME @BenchArgs
    }

    "quick" {
        $BenchArgs = @(
            "python", "/app/benchmark_comparative.py",
            "--image", "/app/test_image.jpg",
            "--runs",  "2",
            "--workers", "1", "4"
        )
        if ($SkipShm)   { $BenchArgs += "--skip-shm" }
        if ($SkipMproc) { $BenchArgs += "--skip-multiprocessing" }
        if ($SaveJson)  { $BenchArgs += @("--output", "/output/results_quick.json") }

        & docker @DockerFlags $IMAGE_NAME @BenchArgs
    }

    "generate" {
        & docker @DockerFlags $IMAGE_NAME `
            python /app/generate_test_image.py --output /app/test_image.jpg --size 512x512
    }

    "shell" {
        & docker @DockerFlags "-it" $IMAGE_NAME bash
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Container exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Done." -ForegroundColor Green
if ($SaveJson) {
    Write-Host "  JSON results saved to: $OutputDir\results.json" -ForegroundColor Green
}
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
