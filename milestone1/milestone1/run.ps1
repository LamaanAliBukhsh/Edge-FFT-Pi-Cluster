# =============================================================================
# Milestone 1 – Build & Run Script (PowerShell)
# Talha Mudassar | PDC Project
# =============================================================================

param(
    [string]$Command  = "run",    # run | benchmark | shell
    [int]   $Threads  = 4,
    [switch]$Naive               # Use pure-Python kernel (shows GIL better)
)

$IMAGE_NAME = "ms1-sobel"
$OUTPUT_DIR = Join-Path $PSScriptRoot "output"

function Build-Image {
    Write-Host "`n[BUILD] Building Docker image '$IMAGE_NAME'..." -ForegroundColor Cyan
    docker build -t $IMAGE_NAME .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Build failed." -ForegroundColor Red; exit 1
    }
    Write-Host "[OK] Image built successfully.`n" -ForegroundColor Green
}

# --------------------------------------------------------------------------
switch ($Command) {

    "run" {
        # Run the edge detector with specified thread count, save output image
        Build-Image
        New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

        $naiveFlag = if ($Naive) { "--naive" } else { "" }

        Write-Host "[RUN] Sobel with $Threads thread(s)..." -ForegroundColor Cyan
        docker run --rm `
            -v "${OUTPUT_DIR}:/output" `
            $IMAGE_NAME `
            python /app/sobel_threaded.py `
                --image /app/test.jpg `
                --threads $Threads `
                --output /output/edges_t${Threads}.png `
                $naiveFlag

        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[OK] Edge image saved to: $OUTPUT_DIR\edges_t${Threads}.png" -ForegroundColor Green
        }
    }

    "benchmark" {
        # Run the full benchmarking suite (all thread counts)
        Build-Image
        $naiveFlag = if ($Naive) { "--naive" } else { "" }

        # Naive (pure-Python) mode is ~50x slower per pixel than vectorized.
        # Under QEMU ARM64 emulation, 512x512 takes 15+ min. Use 64x64 instead;
        # the GIL contention pattern is identical, just completes in seconds.
        $benchImage = if ($Naive) { "/app/test_small.jpg" } else { "/app/test.jpg" }
        $sizeNote   = if ($Naive) { "(64x64  GIL demo mode)" } else { "(512x512)" }
        Write-Host "[BENCHMARK] Image: $sizeNote" -ForegroundColor Yellow
        Write-Host "[BENCHMARK] Running Sobel benchmark (threads: 1, 2, 4, 8)..." -ForegroundColor Cyan
        docker run --rm $IMAGE_NAME `
            python /app/benchmark.py `
                --image $benchImage `
                --runs 3 `
                $naiveFlag
    }

    "shell" {
        # Drop into an interactive shell inside the container
        Build-Image
        Write-Host "[SHELL] Entering container shell..." -ForegroundColor Yellow
        docker run --rm -it $IMAGE_NAME bash
    }

    default {
        Write-Host "Usage: .\run.ps1 [-Command run|benchmark|shell] [-Threads N] [-Naive]"
        Write-Host "  run        - Run Sobel with -Threads N and save output image"
        Write-Host "  benchmark  - Run full benchmark across 1/2/4/8 threads"
        Write-Host "  shell      - Open interactive container shell"
    }
}
