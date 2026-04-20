"""
Comparative Benchmark - Shared Memory vs Threading vs Multiprocessing
======================================================================
Talha Mudassar | PDC Milestone 2

Benchmarks all three single-node parallelism strategies for Sobel edge
detection and produces a unified side-by-side comparison table.

Implementations compared:
  1. threading        - Python threading module (GIL-limited)        [M1 Talha]
  2. multiprocessing  - multiprocessing.Pool (true parallelism)       [M1 Fahad]
  3. shared_memory    - /dev/shm mmap-backed IPC processes            [M2 Muhammad]

Usage:
    python benchmark_comparative.py [--image <path>] [--runs N] [--workers LIST]

Examples:
    python benchmark_comparative.py --image test_image.jpg --runs 3
    python benchmark_comparative.py --image test_image.jpg --runs 5 --workers 1 2 4 8
    python benchmark_comparative.py --image test_image.jpg --runs 3 --output results.json
"""

import argparse
import gc
import json
import multiprocessing as mp
import statistics
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap - allow running from any working directory inside the repo
# ---------------------------------------------------------------------------
_THIS_DIR   = Path(__file__).resolve().parent          # .../milestone2/talha/app/
_REPO_ROOT  = _THIS_DIR.parents[2]                     # .../Edge-FFT-Pi-Cluster/
_M1_THREAD  = _REPO_ROOT / "milestone1" / "milestone1" / "app"
_M1_MPROC   = _REPO_ROOT / "milestone1" / "multiprocessing" / "app"

for _p in [str(_M1_THREAD), str(_M1_MPROC), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import the three implementations
# Threading  (Talha M1)
# ---------------------------------------------------------------------------
try:
    from sobel_threaded import run_sobel_threaded          # type: ignore
    _HAS_THREADING = True
except ImportError as _e:
    print(f"[WARN] Could not import threading implementation: {_e}")
    _HAS_THREADING = False

# ---------------------------------------------------------------------------
# Multiprocessing  (Fahad M1)
# ---------------------------------------------------------------------------
try:
    from sobel_multiprocessing import run_sobel_multiprocessing  # type: ignore
    _HAS_MULTIPROCESSING = True
except ImportError as _e:
    print(f"[WARN] Could not import multiprocessing implementation: {_e}")
    _HAS_MULTIPROCESSING = False

# ---------------------------------------------------------------------------
# Shared Memory IPC  (Muhammad M2)
# ---------------------------------------------------------------------------
try:
    from shared_memory_sobel import SobelEdgeDetector      # type: ignore
    _HAS_SHARED_MEMORY = True
except ImportError as _e:
    print(f"[WARN] Could not import shared_memory implementation: {_e}")
    _HAS_SHARED_MEMORY = False

# ===========================================================================
# Per-method benchmark runners
# ===========================================================================

def _bench_threading(image_path: str, worker_counts: List[int], runs: int) -> Dict[int, float]:
    """Run threading benchmark for each worker count; return {n: avg_time}."""
    if not _HAS_THREADING:
        return {}
    results: Dict[int, float] = {}
    for n in worker_counts:
        times = []
        for _ in range(runs):
            t = run_sobel_threaded(
                image_path=image_path,
                n_threads=n,
                output_path=None,
                use_vectorized=True,
            )
            times.append(t)
        results[n] = statistics.mean(times)
    return results


def _bench_multiprocessing(image_path: str, worker_counts: List[int], runs: int) -> Dict[int, float]:
    """Run multiprocessing benchmark for each worker count; return {n: avg_time}."""
    if not _HAS_MULTIPROCESSING:
        return {}
    results: Dict[int, float] = {}
    for n in worker_counts:
        times = []
        for _ in range(runs):
            t = run_sobel_multiprocessing(
                image_path=image_path,
                n_processes=n,
                output_path=None,
                use_vectorized=True,
            )
            times.append(t)
        results[n] = statistics.mean(times)
    return results


def _bench_shared_memory(image: np.ndarray, worker_counts: List[int], runs: int) -> Dict[int, float]:
    """Run shared-memory benchmark for each worker count; return {n: avg_time}."""
    if not _HAS_SHARED_MEMORY:
        return {}
    results: Dict[int, float] = {}
    for n in worker_counts:
        times = []
        for _ in range(runs):
            detector = SobelEdgeDetector(num_processes=n, use_shm=True)
            _, elapsed = detector.detect_edges_shm(image)
            times.append(elapsed)
            gc.collect()
        results[n] = statistics.mean(times)
    return results


# ===========================================================================
# Display helpers
# ===========================================================================

def _speedup(baseline: float, time: float) -> float:
    return baseline / time if time > 0 else float("inf")


def _efficiency(speedup: float, workers: int) -> float:
    return (speedup / workers) * 100.0


def print_header(image_path: str, runs: int, worker_counts: List[int]) -> None:
    w = 88
    print("\n" + "=" * w)
    print("  COMPARATIVE BENCHMARK - Shared Memory vs Threading vs Multiprocessing")
    print("  Talha Mudassar | PDC Milestone 2")
    print(f"  Image : {image_path}")
    print(f"  Runs  : {runs} per configuration")
    print(f"  Workers tested : {worker_counts}")
    print("=" * w)


def print_comparison_table(
    worker_counts: List[int],
    thread_res: Dict[int, float],
    mproc_res: Dict[int, float],
    shm_res: Dict[int, float],
) -> None:
    """Print the three-way comparison table side-by-side."""

    def _get_baseline(res: Dict[int, float]) -> float:
        if 1 in res:
            return res[1]
        if res:
            return min(res.values())
        return 1.0

    t_base   = _get_baseline(thread_res)
    m_base   = _get_baseline(mproc_res)
    s_base   = _get_baseline(shm_res)

    col = 14  # column width

    sep = "-" * 88
    hdr = (
        f"  {'Workers':>7} | "
        f"{'Threading':>{col}} {'Speedup':>8} | "
        f"{'Multiproc':>{col}} {'Speedup':>8} | "
        f"{'SharedMem':>{col}} {'Speedup':>8}"
    )

    print(f"\n{'Threading Info':^88}")
    print(f"  Single-thread GIL-limited parallelism (M1 - Talha)")
    print(f"\n{'Multiprocessing Info':^88}")
    print(f"  True process-level parallelism via Pool (M1 - Muhammad Fahad)")
    print(f"\n{'Shared Memory Info':^88}")
    print(f"  mmap / /dev/shm IPC processes (M2 - Muhammad)")
    print()
    print(sep)
    print(hdr)
    print(sep)

    for n in worker_counts:
        t_time = thread_res.get(n)
        m_time = mproc_res.get(n)
        s_time = shm_res.get(n)

        def fmt_cell(time_val, base):
            if time_val is None:
                return f"{'N/A':>{col}}  {'N/A':>7} "
            sp = _speedup(base, time_val)
            return f"{time_val:>{col}.4f}  {sp:>6.3f}x "

        print(
            f"  {n:>7} | "
            f"{fmt_cell(t_time, t_base)}| "
            f"{fmt_cell(m_time, m_base)}| "
            f"{fmt_cell(s_time, s_base)}"
        )

    print(sep)


def print_efficiency_table(
    worker_counts: List[int],
    thread_res: Dict[int, float],
    mproc_res: Dict[int, float],
    shm_res: Dict[int, float],
) -> None:
    """Print parallel efficiency for each method."""

    def _base(res):
        return res.get(1, min(res.values()) if res else 1.0)

    t_base = _base(thread_res)
    m_base = _base(mproc_res)
    s_base = _base(shm_res)

    sep = "-" * 60
    print(f"\n  PARALLEL EFFICIENCY  (Efficiency = Speedup / Workers x 100%)")
    print(sep)
    print(f"  {'Workers':>7} | {'Threading':>12} | {'Multiproc':>12} | {'SharedMem':>12}")
    print(sep)

    for n in worker_counts:
        def eff(res, base):
            if n not in res:
                return "      N/A"
            sp = _speedup(base, res[n])
            return f"{_efficiency(sp, n):>10.1f}%"

        print(
            f"  {n:>7} | {eff(thread_res, t_base):>12} | "
            f"{eff(mproc_res, m_base):>12} | {eff(shm_res, s_base):>12}"
        )

    print(sep)


def print_winner_analysis(
    worker_counts: List[int],
    thread_res: Dict[int, float],
    mproc_res: Dict[int, float],
    shm_res: Dict[int, float],
) -> None:
    """Print a brief analysis of which method wins at each worker count."""
    print(f"\n  WINNER ANALYSIS  (fastest method per worker count)")
    print("-" * 60)
    for n in worker_counts:
        contenders = {}
        if n in thread_res:
            contenders["Threading"]      = thread_res[n]
        if n in mproc_res:
            contenders["Multiproc"]      = mproc_res[n]
        if n in shm_res:
            contenders["SharedMem"]      = shm_res[n]
        if not contenders:
            continue
        winner = min(contenders, key=contenders.get)
        winner_time = contenders[winner]
        print(f"  Workers={n:>2}  ->  {winner:<14}  ({winner_time:.4f}s)")
    print("-" * 60)


def print_theoretical_comparison(
    worker_counts: List[int],
    shm_res: Dict[int, float],
) -> None:
    """
    Compare measured shared-memory speedup against Lamaan's theoretical
    predictions (from milestone2/theoretical_analysis/README.md).
    """
    theory = {1: 1.000, 2: 1.602, 4: 2.232, 8: 2.666}  # Amdahl @ f=0.736
    shm_base = shm_res.get(1, None)
    if shm_base is None:
        return

    print(f"\n  THEORETICAL vs MEASURED (SharedMem vs Lamaan's Amdahl f=0.736 predictions)")
    print("-" * 70)
    print(f"  {'Workers':>7} | {'Theory S(P)':>14} | {'Measured S(P)':>14} | {'Delta':>12}")
    print("-" * 70)
    for n in worker_counts:
        if n not in shm_res:
            continue
        measured  = _speedup(shm_base, shm_res[n])
        predicted = theory.get(n, None)
        if predicted is None:
            delta_str = "     N/A"
        else:
            delta = measured - predicted
            delta_str = f"{delta:>+10.3f}x"
        pred_str = f"{predicted:.3f}x" if predicted else "   N/A"
        print(
            f"  {n:>7} | {pred_str:>14} | {measured:>13.3f}x | {delta_str:>12}"
        )
    print("-" * 70)
    print("  Note: Theory is Amdahl single-node ceiling; measured uses /dev/shm IPC.")
    print("        Positive delta = better than predicted; negative = overhead exceeded.")


# ===========================================================================
# JSON export
# ===========================================================================

def save_results_json(
    image_path: str,
    runs: int,
    worker_counts: List[int],
    thread_res: Dict[int, float],
    mproc_res: Dict[int, float],
    shm_res: Dict[int, float],
    output_path: str,
) -> None:
    """Write machine-readable results for Sharjeel's validation report."""
    data = {
        "metadata": {
            "author": "Talha Mudassar",
            "milestone": 2,
            "description": "Comparative benchmark: shared_memory vs threading vs multiprocessing",
            "image": str(image_path),
            "runs_per_config": runs,
            "worker_counts": worker_counts,
        },
        "results": {
            "threading":       {str(k): v for k, v in thread_res.items()},
            "multiprocessing": {str(k): v for k, v in mproc_res.items()},
            "shared_memory":   {str(k): v for k, v in shm_res.items()},
        },
        "speedup": {},
    }

    # Compute speedups
    for method, res in [
        ("threading", thread_res),
        ("multiprocessing", mproc_res),
        ("shared_memory", shm_res),
    ]:
        base = res.get(1, min(res.values()) if res else 1.0)
        data["speedup"][method] = {
            str(k): round(_speedup(base, v), 4) for k, v in res.items()
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"\n  [OK] Results saved -> {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Comparative benchmark: shared_memory vs threading vs multiprocessing\n"
            "Talha Mudassar | PDC Milestone 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="test_image.jpg",
        help="Path to input image (default: test_image.jpg)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Worker counts to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results (for Sharjeel's validation)",
    )
    parser.add_argument(
        "--skip-threading",
        action="store_true",
        help="Skip threading benchmark",
    )
    parser.add_argument(
        "--skip-multiprocessing",
        action="store_true",
        help="Skip multiprocessing benchmark",
    )
    parser.add_argument(
        "--skip-shm",
        action="store_true",
        help="Skip shared-memory benchmark",
    )
    args = parser.parse_args()

    image_path   = args.image
    runs         = args.runs
    worker_counts = sorted(set(args.workers))

    if not Path(image_path).exists():
        print(f"[ERROR] Image not found: {image_path}")
        print("  Run: python generate_test_image.py --output test_image.jpg")
        sys.exit(1)

    # Pre-load the image as a numpy array for shared-memory path
    img_pil  = Image.open(image_path).convert("L")
    img_np   = np.array(img_pil, dtype=np.uint8)

    print_header(image_path, runs, worker_counts)

    # -----------------------------------------------------------------------
    # 1. Threading
    # -----------------------------------------------------------------------
    thread_res: Dict[int, float] = {}
    if not args.skip_threading and _HAS_THREADING:
        print(f"\n  [1/3] Threading benchmark  ({runs} runs x {worker_counts} workers)")
        for n in worker_counts:
            times = []
            for r in range(runs):
                t = run_sobel_threaded(
                    image_path=image_path,
                    n_threads=n,
                    output_path=None,
                    use_vectorized=True,
                )
                times.append(t)
                print(f"        threads={n} run={r+1}/{runs}: {t:.4f}s")
            avg = statistics.mean(times)
            thread_res[n] = avg
            print(f"        threads={n} AVG: {avg:.4f}s")
    elif args.skip_threading:
        print("\n  [1/3] Threading: skipped (--skip-threading)")
    else:
        print("\n  [1/3] Threading: unavailable (import failed)")

    # -----------------------------------------------------------------------
    # 2. Multiprocessing
    # -----------------------------------------------------------------------
    mproc_res: Dict[int, float] = {}
    if not args.skip_multiprocessing and _HAS_MULTIPROCESSING:
        print(f"\n  [2/3] Multiprocessing benchmark  ({runs} runs x {worker_counts} workers)")
        for n in worker_counts:
            times = []
            for r in range(runs):
                t = run_sobel_multiprocessing(
                    image_path=image_path,
                    n_processes=n,
                    output_path=None,
                    use_vectorized=True,
                )
                times.append(t)
                print(f"        procs={n} run={r+1}/{runs}: {t:.4f}s")
            avg = statistics.mean(times)
            mproc_res[n] = avg
            print(f"        procs={n} AVG: {avg:.4f}s")
    elif args.skip_multiprocessing:
        print("\n  [2/3] Multiprocessing: skipped (--skip-multiprocessing)")
    else:
        print("\n  [2/3] Multiprocessing: unavailable (import failed)")

    # -----------------------------------------------------------------------
    # 3. Shared Memory
    # -----------------------------------------------------------------------
    shm_res: Dict[int, float] = {}
    if not args.skip_shm and _HAS_SHARED_MEMORY:
        print(f"\n  [3/3] Shared Memory benchmark  ({runs} runs x {worker_counts} workers)")
        for n in worker_counts:
            times = []
            for r in range(runs):
                detector = SobelEdgeDetector(num_processes=n, use_shm=True)
                _, elapsed = detector.detect_edges_shm(img_np)
                times.append(elapsed)
                gc.collect()
                print(f"        procs={n} run={r+1}/{runs}: {elapsed:.4f}s")
            avg = statistics.mean(times)
            shm_res[n] = avg
            print(f"        procs={n} AVG: {avg:.4f}s")
    elif args.skip_shm:
        print("\n  [3/3] Shared Memory: skipped (--skip-shm)")
    else:
        print("\n  [3/3] Shared Memory: unavailable (import failed)")

    # -----------------------------------------------------------------------
    # Results output
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 88)
    print("  RESULTS")
    print("=" * 88)

    print_comparison_table(worker_counts, thread_res, mproc_res, shm_res)
    print_efficiency_table(worker_counts, thread_res, mproc_res, shm_res)
    print_winner_analysis(worker_counts, thread_res, mproc_res, shm_res)
    print_theoretical_comparison(worker_counts, shm_res)

    # -----------------------------------------------------------------------
    # JSON export (for Sharjeel)
    # -----------------------------------------------------------------------
    if args.output:
        save_results_json(
            image_path=image_path,
            runs=runs,
            worker_counts=worker_counts,
            thread_res=thread_res,
            mproc_res=mproc_res,
            shm_res=shm_res,
            output_path=args.output,
        )

    print("\n" + "=" * 88)
    print("  Benchmark complete.")
    print("  Author : Talha Mudassar | PDC Milestone 2")
    print("  Theory : See milestone2/theoretical_analysis/ (Lamaan Ali Bukhsh)")
    print("  IPC    : See shared_memory_sobel.py (Muhammad Fahad)")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    # Guard required for multiprocessing on Windows
    mp.freeze_support()
    main()
