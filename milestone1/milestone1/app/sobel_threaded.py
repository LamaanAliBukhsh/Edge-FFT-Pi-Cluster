"""
Milestone 1 – Multi-threaded Sobel Edge Detector
==================================================
Talha Mudassar

Implements the Sobel edge detection algorithm using Python's threading module.
The image is split into horizontal strips (one per thread). Each thread computes
the Sobel gradient magnitude for its strip and writes results into a shared
output array, protected by a threading.Lock.

Usage:
    python sobel_threaded.py --image <path> [--threads N] [--output <path>]

Example:
    python sobel_threaded.py --image /app/test_image.jpg --threads 4 --output /app/edges.png
"""

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Sobel kernels (3×3)
# ---------------------------------------------------------------------------
Gx = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)

Gy = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core per-thread worker
# ---------------------------------------------------------------------------
def sobel_worker(gray: np.ndarray,
                 output: np.ndarray,
                 row_start: int,
                 row_end: int,
                 lock: threading.Lock) -> None:
    """
    Compute Sobel gradient magnitude for rows [row_start, row_end) of `gray`.
    Writes the result into the corresponding rows of `output`.

    Parameters
    ----------
    gray      : full-image grayscale array (H × W), float32, range [0, 255]
    output    : shared output array same shape as gray
    row_start : first row this thread is responsible for (inclusive)
    row_end   : last row this thread is responsible for (exclusive)
    lock      : threading.Lock protecting writes to `output`
    """
    H, W = gray.shape
    # Local buffer for this strip's result
    local = np.zeros((row_end - row_start, W), dtype=np.float32)

    for i, global_row in enumerate(range(row_start, row_end)):
        for col in range(W):
            gx_val = 0.0
            gy_val = 0.0
            for ki in range(-1, 2):          # kernel row offset
                for kj in range(-1, 2):      # kernel col offset
                    r = global_row + ki
                    c = col + kj
                    # Clamp to image boundaries (replicate padding)
                    r = max(0, min(H - 1, r))
                    c = max(0, min(W - 1, c))
                    pixel = gray[r, c]
                    gx_val += Gx[ki + 1, kj + 1] * pixel
                    gy_val += Gy[ki + 1, kj + 1] * pixel
            local[i, col] = math.sqrt(gx_val ** 2 + gy_val ** 2)

    # Write local buffer into shared output (lock to prevent data races)
    with lock:
        output[row_start:row_end, :] = local


# ---------------------------------------------------------------------------
# Vectorized per-thread worker (faster, uses numpy for inner loops)
# ---------------------------------------------------------------------------
def sobel_worker_vectorized(gray: np.ndarray,
                             output: np.ndarray,
                             row_start: int,
                             row_end: int,
                             lock: threading.Lock) -> None:
    """
    Vectorized version using numpy convolution for speed.
    Semantically identical to sobel_worker but much faster for larger images.
    """
    from scipy.ndimage import convolve

    strip = gray[max(0, row_start - 1):min(gray.shape[0], row_end + 1), :]
    offset = 1 if row_start > 0 else 0

    gx_conv = convolve(strip, Gx, mode='reflect')
    gy_conv = convolve(strip, Gy, mode='reflect')
    magnitude = np.sqrt(gx_conv ** 2 + gy_conv ** 2)

    local = magnitude[offset:offset + (row_end - row_start), :]

    with lock:
        output[row_start:row_end, :] = local


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_sobel_threaded(image_path: str,
                       n_threads: int = 4,
                       output_path: str = None,
                       use_vectorized: bool = True) -> float:
    """
    Load image, run threaded Sobel, optionally save output.

    Returns
    -------
    elapsed : float  – wall-clock seconds for the threaded computation
    """
    # -- Load & convert to grayscale -----------------------------------------
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)
    H, W = gray.shape

    # -- Prepare shared output & lock ----------------------------------------
    output = np.zeros_like(gray)
    lock = threading.Lock()

    # -- Partition rows into strips ------------------------------------------
    strip_size = H // n_threads
    strips = []
    for t in range(n_threads):
        r_start = t * strip_size
        r_end = H if t == n_threads - 1 else (t + 1) * strip_size
        strips.append((r_start, r_end))

    worker_fn = sobel_worker_vectorized if use_vectorized else sobel_worker

    # -- Launch threads -------------------------------------------------------
    threads = []
    t_start = time.perf_counter()

    for r_start, r_end in strips:
        t = threading.Thread(
            target=worker_fn,
            args=(gray, output, r_start, r_end, lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t_start

    # -- Normalize & save -----------------------------------------------------
    if output is not None:
        out_normalized = np.clip(output, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(out_normalized, mode="L")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path)
            print(f"[Sobel] Edge image saved → {output_path}")

    return elapsed


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-threaded Sobel edge detector (Milestone 1 – Talha Mudassar)"
    )
    parser.add_argument("--image",   required=True,         help="Path to input image")
    parser.add_argument("--threads", type=int, default=4,   help="Number of threads (default: 4)")
    parser.add_argument("--output",  default="/app/edges.png", help="Path to save output edge image")
    parser.add_argument("--naive",   action="store_true",   help="Use pure-Python (slow) kernel instead of numpy")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Milestone 1 – Multi-threaded Sobel Edge Detector")
    print(f"  Talha Mudassar | Parallel & Distributed Computing")
    print(f"{'='*55}")
    print(f"  Image   : {args.image}")
    print(f"  Threads : {args.threads}")
    print(f"  Mode    : {'Pure-Python kernel' if args.naive else 'NumPy vectorized kernel'}")
    print(f"{'='*55}\n")

    elapsed = run_sobel_threaded(
        image_path=args.image,
        n_threads=args.threads,
        output_path=args.output,
        use_vectorized=not args.naive,
    )

    print(f"  ✓ Sobel completed in {elapsed:.4f}s with {args.threads} thread(s)\n")


if __name__ == "__main__":
    main()
