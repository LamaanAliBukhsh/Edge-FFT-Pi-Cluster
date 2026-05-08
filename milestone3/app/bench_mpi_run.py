"""
Milestone 3 - Single-process worker invoked by ``benchmark_mpi.py``
====================================================================

The benchmark harness spawns ``mpirun`` once per ``(P, image, run)`` triple
and reads JSON-encoded timings from this script's stdout.  Keeping the
harness logic separate from the timing kernel avoids polluting the timing
window with subprocess launch costs.

Each invocation:
    1. Loads the requested image on rank 0.
    2. Runs ``equalize_distributed`` ``--warmup + --runs`` times.
    3. On rank 0 prints a single-line JSON record with mean/std/per-run
       wall-clock times and per-phase breakdowns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from histogram_eq_mpi import equalize_distributed  # noqa: E402
from histogram_eq_serial import (  # noqa: E402
    equalize_serial,
    load_grayscale,
)

from mpi4py import MPI  # noqa: E402


def _time_distributed_run(comm: "MPI.Intracomm", image: np.ndarray | None) -> tuple[float, dict]:
    comm.Barrier()
    t0 = MPI.Wtime()
    _, timings = equalize_distributed(comm, image)
    comm.Barrier()
    elapsed = MPI.Wtime() - t0
    return elapsed, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    image: np.ndarray | None = None
    if rank == 0:
        image = load_grayscale(args.image)
        height, width = image.shape
    else:
        height = width = -1

    # warmup runs are not recorded -- they prime the page cache, JIT, and
    # MPI connection lazy setup so the recorded measurements are stable.
    for _ in range(max(0, args.warmup)):
        _time_distributed_run(comm, image)

    durations: list[float] = []
    last_timings: dict = {}
    for _ in range(max(1, args.runs)):
        elapsed, last_timings = _time_distributed_run(comm, image)
        durations.append(elapsed)

    if rank == 0:
        # Cross-check correctness once per (P, image) configuration so we
        # never publish a benchmark number for a broken pipeline.
        reference = equalize_serial(image)  # type: ignore[arg-type]
        result, _ = equalize_distributed(comm, image)
        validated = bool(result is not None and np.array_equal(reference, result))

        record = {
            "label": args.label,
            "image": args.image,
            "height": int(height),
            "width": int(width),
            "world_size": int(size),
            "warmup": int(args.warmup),
            "runs": int(args.runs),
            "mean_s": float(statistics.fmean(durations)),
            "std_s": float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0,
            "min_s": float(min(durations)),
            "max_s": float(max(durations)),
            "per_run_s": [float(d) for d in durations],
            "phases_last_run_s": {
                key: float(value)
                for key, value in last_timings.items()
                if key.endswith("_s")
            },
            "row_counts": list(last_timings.get("row_counts", [])),
            "validated": validated,
        }
        # Single-line JSON makes parsing in the parent harness trivial.
        print("BENCH_RECORD " + json.dumps(record), flush=True)
    else:
        # Non-root ranks must still participate in the validation re-run so
        # the collective calls match across the communicator.
        equalize_distributed(comm, image)

    return 0


if __name__ == "__main__":
    sys.exit(main())
