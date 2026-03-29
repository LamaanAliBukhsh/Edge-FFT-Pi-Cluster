FROM pdcm0-base:latest

# Milestone 1 – Multiprocessing Sobel Implementation
# Muhammad Fahad

WORKDIR /app

# Copy application code
COPY app/ /app/

# Generate test image
RUN python3 generate_test_image.py --output /app/test_image.jpg

# Default command: run help for multiprocessing Sobel
CMD ["python3", "sobel_multiprocessing.py", "--help"]
