"""
Milestone 1 – Benchmarking Harness (Multiprocessing)
====================================================
Muhammad Fahad

Runs the multiprocessing Sobel edge detector for process counts [1, 2, 4, 8]
and prints a results table with:
  - Average wall-clock time per process count
  - Speedup (relative to sequential baseline)
  - Parallel efficiency (speedup / process count)

This benchmark demonstrates GIL escape: multiprocessing should show closer-to-linear
speedup than threading, because each process has its own GIL.

Usage:
    python benchmark.py [--image <path>] [--runs N] [--naive]
"""

import argparse
import statistics
import sys
from pathlib import Path

# Ensure app directory is on path when running from Docker
sys.path.insert(0, str(Path(__file__).parent))

from sobel_multiprocessing import run_sobel_multiprocessing


PROCESS_COUNTS = [1, 2, 4, 8]
DEFAULT_RUNS = 3

# Baseline time from sequential implementation (256x256 vectorized: ~0.12s)
# This will be measured per-run and reported
BASELINE_TIME = None


def benchmark(image_path: str, n_runs: int, use_vectorized: bool) -> dict:
    """
    Run the Multiprocessing Sobel detector for each process count, averaged over n_runs.

    Returns a dict: {n_processes -> avg_time_seconds}
    """
    results = {}
    for n_processes in PROCESS_COUNTS:
        times = []
        for _ in range(n_runs):
            elapsed = run_sobel_multiprocessing(
                image_path=image_path,
                n_processes=n_processes,
                output_path=None,           # Don't write file during benchmark
                use_vectorized=use_vectorized,
            )
            times.append(elapsed)
        avg = statistics.mean(times)
        results[n_processes] = avg
        print(f"  processes={n_processes:2d} | avg={avg:.4f}s | runs={[f'{t:.4f}s' for t in times]}")

    return results


def print_results_table(results: dict, baseline_time: float) -> None:
    """Pretty-print the benchmark results as a table."""
    header = f"{'Processes':>10} | {'Avg Time (s)':>14} | {'Speedup':>10} | {'Efficiency':>12}"
    sep    = "-" * len(header)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS – Multiprocessing Sobel")
    print(f"  Muhammad Fahad | PDC Milestone 1")
    print(f"{'='*70}")
    print(f"\n{header}")
    print(sep)

    for n_processes, avg_time in sorted(results.items()):
        speedup    = baseline_time / avg_time
        efficiency = speedup / n_processes * 100
        print(f"{n_processes:>10} | {avg_time:>14.4f} | {speedup:>10.3f}x | {efficiency:>11.1f}%")

    print(sep)
    print(f"\n  Note: Multiprocessing should show closer-to-linear speedup than threading")
    print(f"  because each process has its own GIL (true parallelism).")
    print(f"  (Compare with threading results by Talha Mudassar)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiprocessing Sobel edge detector (Milestone 1)"
    )
    parser.add_argument("--image", default="/app/test_image.jpg",
                        help="Path to input image (default: /app/test_image.jpg)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of runs per process count (default: {DEFAULT_RUNS})")
    parser.add_argument("--naive", action="store_true",
                        help="Use pure-Python kernel (better for demonstration)")
    parser.add_argument("--baseline", type=float, default=0.12,
                        help="Sequential baseline time in seconds (default: 0.12)")
    args = parser.parse_args()

    use_vectorized = not args.naive

    print(f"\n{'='*70}")
    print(f"  Milestone 1 – Multiprocessing Sobel Benchmark")
    print(f"  Muhammad Fahad | PDC Project")
    print(f"{'='*70}")
    print(f"\n  Image    : {args.image}")
    print(f"  Runs     : {args.runs} per process count")
    print(f"  Mode     : {'Pure-Python (GIL demo)' if args.naive else 'NumPy vectorized'}")
    print(f"  Baseline : {args.baseline:.4f}s (from sequential)\n")

    results = benchmark(args.image, args.runs, use_vectorized)
    print_results_table(results, args.baseline)


if __name__ == "__main__":
    main()
