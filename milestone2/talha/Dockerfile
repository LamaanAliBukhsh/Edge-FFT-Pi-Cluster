# syntax=docker/dockerfile:1
# ============================================================
#  Milestone 2 – Comparative Benchmark Container
#  Talha Mudassar | PDC Spring 2025
#
#  Bundles:
#    - Talha's M2 benchmark script
#    - Talha's M1 threaded Sobel
#    - Fahad's M1 multiprocessing Sobel
#    - Muhammad's M2 shared_memory_sobel.py
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Talha Mudassar"
LABEL milestone="2"
LABEL description="Comparative benchmark: shared_memory vs threading vs multiprocessing"

# System deps (none beyond pip packages needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python deps shared across all three implementations
RUN pip install --no-cache-dir numpy scipy pillow

# ---------------------------------------------------------------------------
# Copy implementation sources
# (Build context = repo root; all paths relative to repo root)
# ---------------------------------------------------------------------------

# M1 – Threading (Talha)
COPY milestone1/milestone1/app/ /milestone1/milestone1/app/

# M1 – Multiprocessing (Fahad)
COPY milestone1/multiprocessing/app/ /milestone1/multiprocessing/app/

# M2 – Shared memory IPC (Muhammad) – root-level module
COPY shared_memory_sobel.py /shared_memory_sobel.py

# M2 – This benchmark
COPY milestone2/talha/app/ /app/

# ---------------------------------------------------------------------------
# Generate test image at build time so container is self-contained
# ---------------------------------------------------------------------------
RUN python /app/generate_test_image.py --output /app/test_image.jpg --size 512x512

# ---------------------------------------------------------------------------
# Runtime environment variables
# ---------------------------------------------------------------------------
ENV PYTHONPATH="/milestone1/milestone1/app:/milestone1/multiprocessing/app:/"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Default: run the full comparative benchmark
CMD ["python", "benchmark_comparative.py", \
     "--image", "/app/test_image.jpg", \
     "--runs", "3", \
     "--workers", "1", "2", "4", "8"]
