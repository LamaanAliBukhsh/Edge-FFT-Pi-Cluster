"""
Milestone 1 – Sequential (Single-Core) Sobel Edge Detector
===========================================================
Lamaan Ali Bukhsh

Implements the Sobel edge detection algorithm using pure sequential computation.
No threading, no parallelism. This is the baseline for speedup measurements.

Usage:
    python sobel_sequential.py --image <path> [--output <path>]

Example:
    python sobel_sequential.py --image /app/test_image.jpg --output /app/edges.png
"""

import argparse
import math
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
# Sequential (pure-Python) implementation
# ---------------------------------------------------------------------------
def sobel_sequential_naive(gray: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient magnitude using pure Python loops.
    Baseline implementation—no vectorization, no parallelism.

    Parameters
    ----------
    gray : ndarray
        Grayscale image (H × W), float32, range [0, 255]

    Returns
    -------
    edges : ndarray
        Edge magnitude image, same shape as gray
    """
    H, W = gray.shape
    output = np.zeros_like(gray)

    for i in range(H):
        for j in range(W):
            gx_val = 0.0
            gy_val = 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    r = max(0, min(H - 1, i + ki))
                    c = max(0, min(W - 1, j + kj))
                    pixel = gray[r, c]
                    gx_val += Gx[ki + 1, kj + 1] * pixel
                    gy_val += Gy[ki + 1, kj + 1] * pixel
            output[i, j] = math.sqrt(gx_val ** 2 + gy_val ** 2)

    return output


# ---------------------------------------------------------------------------
# Vectorized sequential implementation (faster, still single-core)
# ---------------------------------------------------------------------------
def sobel_sequential_vectorized(gray: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient magnitude using NumPy convolution.
    Fast sequential implementation using NumPy (single-core optimization).
    """
    from scipy.ndimage import convolve

    gx_conv = convolve(gray, Gx, mode='reflect')
    gy_conv = convolve(gray, Gy, mode='reflect')
    magnitude = np.sqrt(gx_conv ** 2 + gy_conv ** 2)

    return magnitude


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_sobel_sequential(image_path: str,
                         output_path: str = None,
                         use_vectorized: bool = True) -> float:
    """
    Load image, run sequential Sobel, optionally save output.

    Returns
    -------
    elapsed : float
        Wall-clock seconds for the Sobel computation (excludes I/O)
    """
    # -- Load & convert to grayscale -----------------------------------------
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)

    print(f"[*] Loaded image: {gray.shape} (H × W)")
    print(f"[*] Mode: {'Vectorized (NumPy)' if use_vectorized else 'Pure-Python'}")

    # -- Run Sobel computation -----------------------------------------------
    sobel_fn = sobel_sequential_vectorized if use_vectorized else sobel_sequential_naive

    t_start = time.perf_counter()
    edges = sobel_fn(gray)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    print(f"[*] Sobel computation: {elapsed:.4f}s")

    # -- Normalize to [0, 255] for display -----------------------------------
    edges_normalized = ((edges / edges.max()) * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)

    # -- Optionally save output ----------------------------------------------
    if output_path:
        output_img = Image.fromarray(edges_normalized, mode="L")
        output_img.save(output_path)
        print(f"[+] Saved edge image → {output_path}")

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Sequential (single-core) Sobel edge detector – Milestone 1 baseline"
    )
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    parser.add_argument("--output", default=None,
                        help="Path to save output edge image")
    parser.add_argument("--naive", action="store_true",
                        help="Use pure-Python kernel (slower but shows baseline clearly)")
    args = parser.parse_args()

    use_vectorized = not args.naive
    elapsed = run_sobel_sequential(
        image_path=args.image,
        output_path=args.output,
        use_vectorized=use_vectorized,
    )

    print(f"\n[RESULT] Sequential Sobel: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
