"""
Milestone 1 – Benchmarking Harness
====================================
Talha Mudassar

Runs the multi-threaded Sobel edge detector for thread counts [1, 2, 4, 8]
and prints a results table with:
  - Average wall-clock time per thread count
  - Speedup relative to 1 thread
  - Parallel efficiency

This script is designed to feed results into Sharjeel's GIL analysis (benchmarking task).

Usage:
    python benchmark.py [--image <path>] [--runs N] [--naive]
"""

import argparse
import statistics
import sys
from pathlib import Path

# Ensure app directory is on path when running from Docker
sys.path.insert(0, str(Path(__file__).parent))

from sobel_threaded import run_sobel_threaded


THREAD_COUNTS = [1, 2, 4, 8]
DEFAULT_RUNS   = 3


def benchmark(image_path: str, n_runs: int, use_vectorized: bool) -> dict:
    """
    Run the Sobel detector for each thread count, averaged over n_runs.

    Returns a dict: {n_threads -> avg_time_seconds}
    """
    results = {}
    for n_threads in THREAD_COUNTS:
        times = []
        for _ in range(n_runs):
            elapsed = run_sobel_threaded(
                image_path=image_path,
                n_threads=n_threads,
                output_path=None,           # Don't write file during benchmark
                use_vectorized=use_vectorized,
            )
            times.append(elapsed)
        avg = statistics.mean(times)
        results[n_threads] = avg
        print(f"  threads={n_threads:2d} | avg={avg:.4f}s | runs={times}")

    return results


def print_results_table(results: dict) -> None:
    """Pretty-print the benchmark results as a table."""
    baseline = results[1]

    header = f"{'Threads':>8} | {'Avg Time (s)':>14} | {'Speedup':>10} | {'Efficiency':>12}"
    sep    = "-" * len(header)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS – Multi-threaded Sobel (threading)")
    print(f"  Talha Mudassar | PDC Milestone 1")
    print(f"{'='*60}")
    print(f"\n{header}")
    print(sep)

    for n_threads, avg_time in sorted(results.items()):
        speedup    = baseline / avg_time
        efficiency = speedup / n_threads * 100
        print(f"{n_threads:>8} | {avg_time:>14.4f} | {speedup:>10.3f}x | {efficiency:>11.1f}%")

    print(sep)
    print(f"\n  Note: Speedup < N threads is expected due to Python's GIL.")
    print(f"  The GIL prevents true parallelism for CPU-bound threading.")
    print(f"  (See multiprocessing version by Muhammad Fahad for comparison)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-threaded Sobel edge detector (Milestone 1)"
    )
    parser.add_argument("--image", default="/app/test_image.jpg",
                        help="Path to input image (default: /app/test_image.jpg)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of runs per thread count (default: {DEFAULT_RUNS})")
    parser.add_argument("--naive", action="store_true",
                        help="Use pure-Python kernel (better for GIL demonstration)")
    args = parser.parse_args()

    use_vectorized = not args.naive

    print(f"\n{'='*60}")
    print(f"  Milestone 1 – Sobel Benchmark")
    print(f"  Image   : {args.image}")
    print(f"  Runs    : {args.runs} per thread count")
    print(f"  Mode    : {'Pure-Python (GIL visible)' if args.naive else 'NumPy vectorized'}")
    print(f"  Testing : threads = {THREAD_COUNTS}")
    print(f"{'='*60}\n")

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        print("  Run 'python generate_test_image.py' to create a test image first.")
        sys.exit(1)

    results = benchmark(args.image, args.runs, use_vectorized)
    print_results_table(results)


if __name__ == "__main__":
    main()
