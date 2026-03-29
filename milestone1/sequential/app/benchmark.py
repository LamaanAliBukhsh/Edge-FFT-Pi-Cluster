"""
Milestone 1 – Benchmarking Harness (Sequential)
===============================================
Lamaan Ali Bukhsh

Runs the sequential Sobel edge detector multiple times and reports:
  - Average wall-clock time
  - Raw times per run
  - This is the BASELINE (always 1.0x speedup)

This baseline is the denominator for all speedup calculations in M1.

Usage:
    python benchmark.py [--image <path>] [--runs N] [--naive]
"""

import argparse
import statistics
import sys
from pathlib import Path

# Ensure app directory is on path when running from Docker
sys.path.insert(0, str(Path(__file__).parent))

from sobel_sequential import run_sobel_sequential


def benchmark(image_path: str, n_runs: int, use_vectorized: bool) -> float:
    """
    Run the Sequential Sobel detector n_runs times.

    Returns
    -------
    avg_time : float
        Average wall-clock seconds across all runs
    """
    times = []
    for run_num in range(n_runs):
        elapsed = run_sobel_sequential(
            image_path=image_path,
            output_path=None,           # Don't write file during benchmark
            use_vectorized=use_vectorized,
        )
        times.append(elapsed)
        print(f"  Run {run_num + 1}/{n_runs}: {elapsed:.4f}s")

    avg = statistics.mean(times)
    print(f"\n  Average: {avg:.4f}s")
    print(f"  Raw times: {[f'{t:.4f}s' for t in times]}")
    return avg


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential Sobel edge detector (Milestone 1 baseline)"
    )
    parser.add_argument("--image", default="/app/test_image.jpg",
                        help="Path to input image (default: /app/test_image.jpg)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs for averaging (default: 3)")
    parser.add_argument("--naive", action="store_true",
                        help="Use pure-Python kernel (slower but baseline-clear)")
    args = parser.parse_args()

    use_vectorized = not args.naive

    print(f"\n{'='*70}")
    print(f"  Milestone 1 – Sequential Sobel Benchmark")
    print(f"  Lamaan Ali Bukhsh | PDC Project")
    print(f"{'='*70}")
    print(f"\n  Image   : {args.image}")
    print(f"  Runs    : {args.runs}")
    print(f"  Mode    : {'Pure-Python (naive)' if args.naive else 'Vectorized (NumPy)'}\n")

    avg_time = benchmark(args.image, args.runs, use_vectorized)

    print(f"\n{'='*70}")
    print(f"  BASELINE RESULT – Sequential Sobel")
    print(f"{'='*70}")
    print(f"  Average Time: {avg_time:.4f}s")
    print(f"  Speedup:      1.000x  (this is the reference baseline)")
    print(f"  Efficiency:   100.0%  (N/A for sequential)")
    print(f"{'='*70}\n")
    print(f"  NOTE: All speedup calculations in M1 use this baseline.")
    print(f"        Threaded speedup = {avg_time:.4f}s / threaded_time")
    print(f"\n")


if __name__ == "__main__":
    main()
