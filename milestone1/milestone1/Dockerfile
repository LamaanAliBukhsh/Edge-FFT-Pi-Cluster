# =============================================================================
# Milestone 1: Single-Node Sobel Edge Detector Container
# Based on Milestone 0 image; adds Pillow and scipy for image processing
# =============================================================================
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. Install system dependencies (keep lean – no MPI needed for single-node MS1)
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

# Pre-generate the main test image (512×512) and a small one (64×64) for naive-mode benchmarks
RUN python /app/generate_test_image.py --size 512 --output /app/test_image.jpg && \
    python /app/generate_test_image.py --size 64  --output /app/test_small.jpg

# Default: run edge detector with 4 threads
CMD ["python", "/app/sobel_threaded.py", "--image", "/app/test.jpg", \
     "--threads", "4", "--output", "/app/edges.png"]

