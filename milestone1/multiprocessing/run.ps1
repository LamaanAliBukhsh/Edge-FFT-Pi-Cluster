# Milestone 1 – Multiprocessing Sobel Launcher
# Muhammad Fahad | Windows PowerShell

param(
    [string]$Command = "benchmark",
    [int]$Runs = 3,
    [switch]$Naive = $false,
    [switch]$Docker = $false
)

function Show-Usage {
    Write-Host @"
USAGE: .\run.ps1 [options]

COMMANDS:
  benchmark   Run benchmark for 1,2,4,8 processes (default)
  test        Run single Sobel computation and save output
  shell       Interactive Python shell in app directory

OPTIONS:
  -Runs <N>     Number of runs per process count (default: 3)
  -Naive        Use pure-Python kernel (--naive flag)
  -Docker       Run in Docker container (requires image build)

EXAMPLES:
  # Quick benchmark (3 runs, vectorized)
  .\run.ps1 benchmark

  # Extended benchmark with pure-Python kernel
  .\run.ps1 benchmark -Runs 5 -Naive

  # Run single computation in Docker
  .\run.ps1 test -Docker

  # Interactive shell
  .\run.ps1 shell
"@
}

function Run-Local {
    Write-Host "MULTIPROCESSING SOBEL – Muhammad Fahad" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Set-Location app
    
    switch ($Command.ToLower()) {
        "benchmark" {
            $args = @("benchmark.py", "--runs", $Runs.ToString())
            if ($Naive) { $args += "--naive" }
            Write-Host "Running: python3 $args"
            python3 @args
        }
        "test" {
            python3 generate_test_image.py --output test_image.jpg
            Write-Host ""
            $args = @("sobel_multiprocessing.py", "--image", "test_image.jpg", "--output", "test_output.jpg", "--n-processes", "2")
            if ($Naive) { $args += "--naive" }
            Write-Host "Running: python3 $args"
            python3 @args
            Write-Host ""
            if (Test-Path test_output.jpg) {
                $size = (Get-Item test_output.jpg).Length
                Write-Host "Output saved: test_output.jpg ($size bytes)" -ForegroundColor Green
            }
        }
        "shell" {
            Write-Host "Entering Python shell in $PWD" -ForegroundColor Yellow
            python3
        }
        default { Show-Usage }
    }
    
    Set-Location ..
}

function Run-Docker {
    Write-Host "Building and running in Docker..." -ForegroundColor Cyan
    
    # Build the image
    Write-Host "Building Docker image: pdcm0-multiprocessing:latest" -ForegroundColor Yellow
    docker build -t pdcm0-multiprocessing:latest -f Dockerfile .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker build failed" -ForegroundColor Red
        exit 1
    }
    
    # Run container based on command
    switch ($Command.ToLower()) {
        "benchmark" {
            $args = @("--rm", "-v", "${PWD}/app:/app", "pdcm0-multiprocessing:latest", "python3", "benchmark.py", "--runs", $Runs.ToString())
            if ($Naive) { $args += "--naive" }
            docker run @args
        }
        "test" {
            docker run --rm -v "${PWD}/app:/app" pdcm0-multiprocessing:latest python3 sobel_multiprocessing.py --image /app/test_image.jpg --output /app/test_output_docker.jpg --n-processes 2
        }
        "shell" {
            docker run --rm -it -v "${PWD}/app:/app" pdcm0-multiprocessing:latest /bin/bash
        }
    }
}

# Main entry point
if ($Command -eq "--help" -or $Command -eq "-h" -or $Command -eq "help") {
    Show-Usage
    exit 0
}

if ($Docker) {
    Run-Docker
} else {
    Run-Local
}
