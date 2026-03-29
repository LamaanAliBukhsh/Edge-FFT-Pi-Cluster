# =============================================================================
# Milestone 1 – Sequential Sobel Build & Run Script (PowerShell)
# Lamaan Ali Bukhsh | PDC Project
# =============================================================================

param(
    [string]$Command  = "run",    # run | benchmark | shell
    [switch]$Naive               # Use pure-Python kernel
)

$IMAGE_NAME = "ms1-sequential"
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
        # Run the sequential edge detector, save output image
        Build-Image
        New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

        $naiveFlag = if ($Naive) { "--naive" } else { "" }

        Write-Host "[RUN] Sequential Sobel on 512×512 image..." -ForegroundColor Cyan
        docker run --rm `
            -v "${OUTPUT_DIR}:/output" `
            $IMAGE_NAME `
            python /app/sobel_sequential.py `
                --image /app/test_image.jpg `
                --output /output/edges_sequential.png `
                $naiveFlag

        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[OK] Edge image saved to: $OUTPUT_DIR\edges_sequential.png" -ForegroundColor Green
        }
    }

    "benchmark" {
        # Run the benchmark suite (sequential baseline)
        Build-Image
        $naiveFlag = if ($Naive) { "--naive" } else { "" }

        $benchImage = if ($Naive) { "/app/test_small.jpg" } else { "/app/test_image.jpg" }
        $sizeNote   = if ($Naive) { "(64x64  naive mode)" } else { "(512x512)" }
        
        Write-Host "[BENCHMARK] Image: $sizeNote" -ForegroundColor Yellow
        Write-Host "[BENCHMARK] Running Sequential Sobel benchmark (3 runs)..." -ForegroundColor Cyan
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
        Write-Host "Usage: .\run.ps1 [-Command run|benchmark|shell] [-Naive]"
        Write-Host "  run        - Run sequential Sobel and save output image"
        Write-Host "  benchmark  - Run benchmark for baseline timing (3 runs)"
        Write-Host "  shell      - Open interactive container shell"
    }
}
