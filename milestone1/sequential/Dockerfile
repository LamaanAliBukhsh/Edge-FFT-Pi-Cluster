# =============================================================================
# Milestone 1: Sequential Sobel Edge Detector Container
# Lamaan Ali Bukhsh | PDC Project
# Single-core baseline (no threading, no multiprocessing, no MPI)
# =============================================================================
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. Install system dependencies (keep lean – sequential only)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Install Python packages
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    Pillow==10.3.0 \
    scipy==1.11.4

# -----------------------------------------------------------------------------
# 3. Copy application code
# -----------------------------------------------------------------------------
WORKDIR /app
COPY app/ /app/

# Pre-generate test images (512×512 and 64×64)
RUN python /app/generate_test_image.py --size 512 --output /app/test_image.jpg && \
    python /app/generate_test_image.py --size 64  --output /app/test_small.jpg

# Default: run sequential Sobel with 512×512 test image
CMD ["python", "/app/sobel_sequential.py", "--image", "/app/test_image.jpg", \
     "--output", "/app/edges.png"]
