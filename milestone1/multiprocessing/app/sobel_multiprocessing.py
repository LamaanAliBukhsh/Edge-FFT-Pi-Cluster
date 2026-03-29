"""
Milestone 1 – Multiprocessing Sobel Edge Detector
==================================================
Muhammad Fahad

Implements the Sobel edge detection algorithm using Python's multiprocessing module.
Each process handles an independent horizontal strip of the image, enabling true
parallelism and circumventing the Global Interpreter Lock (GIL).

Usage:
    python sobel_multiprocessing.py --image <path> [--processes N] [--output <path>]

Example:
    python sobel_multiprocessing.py --image /app/test_image.jpg --processes 4 --output /app/edges.png
"""

import argparse
import math
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Sobel kernels (3x3)
# ---------------------------------------------------------------------------
Gx = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)

Gy = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------
def sobel_worker(args: tuple) -> tuple:
    """
    Compute Sobel gradient magnitude for a horizontal strip of the image.
    This function runs in a separate process, enabling true parallelism.

    Parameters
    ----------
    args : tuple
        (gray, row_start, row_end, use_vectorized)
        - gray: full grayscale image array (H x W)
        - row_start: first row this process handles
        - row_end: last row this process handles
        - use_vectorized: whether to use NumPy or pure-Python

    Returns
    -------
    (row_start, row_end, local_result) : tuple
        - row_start, row_end: row indices (for reassembly)
        - local_result: edge magnitude for this strip
    """
    gray, row_start, row_end, use_vectorized = args

    if use_vectorized:
        # Vectorized path: use scipy for speed (within single process)
        from scipy.ndimage import convolve
        strip = gray[max(0, row_start - 1):min(gray.shape[0], row_end + 1), :]
        offset = 1 if row_start > 0 else 0
        gx_conv = convolve(strip, Gx, mode='reflect')
        gy_conv = convolve(strip, Gy, mode='reflect')
        magnitude = np.sqrt(gx_conv ** 2 + gy_conv ** 2)
        local = magnitude[offset:offset + (row_end - row_start), :]
    else:
        # Pure-Python path (slower, but demonstrates no GIL blocking)
        H, W = gray.shape
        local = np.zeros((row_end - row_start, W), dtype=np.float32)
        for i, global_row in enumerate(range(row_start, row_end)):
            for col in range(W):
                gx_val = 0.0
                gy_val = 0.0
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        r = max(0, min(H - 1, global_row + ki))
                        c = max(0, min(W - 1, col + kj))
                        pixel = gray[r, c]
                        gx_val += Gx[ki + 1, kj + 1] * pixel
                        gy_val += Gy[ki + 1, kj + 1] * pixel
                local[i, col] = math.sqrt(gx_val ** 2 + gy_val ** 2)

    return (row_start, row_end, local)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_sobel_multiprocessing(image_path: str,
                              n_processes: int = 4,
                              output_path: str = None,
                              use_vectorized: bool = True) -> float:
    """
    Load image, run multiprocessing Sobel, optionally save output.

    Returns
    -------
    elapsed : float
        Wall-clock seconds for the multiprocessing computation (excludes I/O)
    """
    # -- Load & convert to grayscale -----------------------------------------
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)
    H, W = gray.shape

    print(f"[*] Loaded image: {gray.shape} (H x W)")
    print(f"[*] Processes: {n_processes}")
    print(f"[*] Mode: {'Vectorized (NumPy)' if use_vectorized else 'Pure-Python'}")

    # -- Prepare output array ------------------------------------------------
    output = np.zeros_like(gray)

    # -- Partition rows into strips ------------------------------------------
    strip_size = H // n_processes
    work_items = []
    for p in range(n_processes):
        r_start = p * strip_size
        r_end = H if p == n_processes - 1 else (p + 1) * strip_size
        work_items.append((gray, r_start, r_end, use_vectorized))

    # -- Launch parallel workers with Pool -----------------------------------
    t_start = time.perf_counter()

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(sobel_worker, work_items)

    # -- Reassemble results from all processes -------------------------------
    for row_start, row_end, local_result in results:
        output[row_start:row_end, :] = local_result

    t_end = time.perf_counter()

    elapsed = t_end - t_start
    print(f"[*] Multiprocessing Sobel computation: {elapsed:.4f}s")

    # -- Normalize to [0, 255] for display -----------------------------------
    edges_normalized = ((output / output.max()) * 255).astype(np.uint8) if output.max() > 0 else output.astype(np.uint8)

    # -- Optionally save output ----------------------------------------------
    if output_path:
        output_img = Image.fromarray(edges_normalized, mode="L")
        output_img.save(output_path)
        print(f"[+] Saved edge image: {output_path}")

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Multiprocessing Sobel edge detector – Milestone 1 (GIL escape test)"
    )
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of worker processes (default: 4)")
    parser.add_argument("--output", default=None,
                        help="Path to save output edge image")
    parser.add_argument("--naive", action="store_true",
                        help="Use pure-Python kernel (slower but shows parallelism)")
    args = parser.parse_args()

    use_vectorized = not args.naive
    elapsed = run_sobel_multiprocessing(
        image_path=args.image,
        n_processes=args.processes,
        output_path=args.output,
        use_vectorized=use_vectorized,
    )

    print(f"\n[RESULT] Multiprocessing Sobel ({args.processes} processes): {elapsed:.4f}s")


if __name__ == "__main__":
    main()
