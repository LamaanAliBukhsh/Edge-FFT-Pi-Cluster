"""
Milestone 3 - MPI benchmark harness (P = 1, 2, 3, 4)
=====================================================

Roadmap M3 task 08 ("Re-Run M1 Benchmarks on Physical Hardware") asks for a
speedup curve at P = 1, 2, 3, 4 against the single-process baseline.  We run
inside the existing 4-container Docker cluster instead of physical Pis (each
container is pinned to one CPU, so adding ranks adds CPUs identically to the
hardware case).

This harness is meant to be invoked **from inside ``n1``** (rank 0 / master)
once the cluster is up.  It:

    * Spawns ``mpirun`` one process group at a time for P in --ranks.
    * Parses the single JSON record emitted by ``bench_mpi_run.py``.
    * Computes speedup vs. P=1 and parallel efficiency.
    * Writes a CSV identical in spirit to the M1 harness so plots can be
      stitched together later in M5.
    * Prints a human-readable summary table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Bench JSON lines are prefixed so we can ignore mpirun's own chatter.
RECORD_PREFIX = "BENCH_RECORD "

DEFAULT_HOSTS = ["n1", "n2", "n3", "n4"]


def _select_hosts(world_size: int, hosts: list[str]) -> str:
    if world_size > len(hosts):
        raise ValueError(f"world_size={world_size} exceeds available hosts ({hosts})")
    return ",".join(hosts[:world_size])


def _run_one(
    image: str,
    world_size: int,
    runs: int,
    warmup: int,
    hosts: list[str],
    worker_script: str,
    extra_mpirun: list[str] | None = None,
    label: str = "",
    python_bin: str = "python",
) -> dict:
    """Spawn one ``mpirun`` group and return the parsed JSON record."""
    host_list = _select_hosts(world_size, hosts)
    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--host", host_list,
        "-np", str(world_size),
        *(extra_mpirun or []),
        python_bin,
        worker_script,
        "--image", image,
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--label", label or f"P{world_size}",
    ]
    print(f"[bench] >>> {shlex.join(cmd)}", flush=True)

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"mpirun (np={world_size}) failed with exit code {completed.returncode}"
        )

    record: dict | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            record = json.loads(line[len(RECORD_PREFIX):])
            break

    if record is None:
        sys.stderr.write(completed.stdout)
        raise RuntimeError("benchmark worker did not emit a BENCH_RECORD line")

    return record


def _print_table(records: list[dict]) -> None:
    header = (
        f"{'P':>3} | {'mean (s)':>10} | {'std (s)':>9} | "
        f"{'speedup':>8} | {'efficiency':>10} | {'validated':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    baseline = next((r["mean_s"] for r in records if r["world_size"] == 1), None)
    for r in records:
        if baseline and baseline > 0:
            speedup = baseline / r["mean_s"]
            efficiency = speedup / r["world_size"] * 100
            speedup_s = f"{speedup:>7.3f}x"
            efficiency_s = f"{efficiency:>9.1f}%"
        else:
            speedup_s = "      -"
            efficiency_s = "        -"
        valid = "YES" if r.get("validated") else "NO"
        print(
            f"{r['world_size']:>3} | {r['mean_s']:>10.4f} | {r['std_s']:>9.4f} | "
            f"{speedup_s} | {efficiency_s} | {valid:>9}"
        )
    print("=" * len(header) + "\n")


def _write_csv(records: list[dict], path: Path, image_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = next((r["mean_s"] for r in records if r["world_size"] == 1), None)
    fieldnames = [
        "image",
        "image_label",
        "world_size",
        "runs",
        "warmup",
        "mean_s",
        "std_s",
        "min_s",
        "max_s",
        "speedup",
        "efficiency",
        "validated",
        "row_counts",
        "scatter_s",
        "local_hist_s",
        "allreduce_s",
        "apply_s",
        "gather_s",
        "total_compute_s",
    ]

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            phases = r.get("phases_last_run_s", {})
            speedup = baseline / r["mean_s"] if baseline else ""
            efficiency = (
                (speedup / r["world_size"]) if isinstance(speedup, float) else ""
            )
            writer.writerow(
                {
                    "image": r["image"],
                    "image_label": image_label,
                    "world_size": r["world_size"],
                    "runs": r["runs"],
                    "warmup": r["warmup"],
                    "mean_s": f"{r['mean_s']:.6f}",
                    "std_s": f"{r['std_s']:.6f}",
                    "min_s": f"{r['min_s']:.6f}",
                    "max_s": f"{r['max_s']:.6f}",
                    "speedup": f"{speedup:.4f}" if isinstance(speedup, float) else "",
                    "efficiency": (
                        f"{efficiency:.4f}" if isinstance(efficiency, float) else ""
                    ),
                    "validated": r.get("validated", False),
                    "row_counts": "|".join(str(c) for c in r.get("row_counts", [])),
                    "scatter_s": f"{phases.get('scatter_s', 0.0):.6f}",
                    "local_hist_s": f"{phases.get('local_hist_s', 0.0):.6f}",
                    "allreduce_s": f"{phases.get('allreduce_s', 0.0):.6f}",
                    "apply_s": f"{phases.get('apply_s', 0.0):.6f}",
                    "gather_s": f"{phases.get('gather_s', 0.0):.6f}",
                    "total_compute_s": (
                        f"{phases.get('total_compute_s', 0.0):.6f}"
                    ),
                }
            )


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speedup benchmark for distributed histogram equalisation"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the test image (must be readable from every container)",
    )
    parser.add_argument(
        "--ranks",
        default="1,2,3,4",
        help="Comma-separated world sizes to sweep (default 1,2,3,4)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed runs per rank count (default 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per rank count, not recorded (default 1)",
    )
    parser.add_argument(
        "--hosts",
        default=",".join(DEFAULT_HOSTS),
        help="Comma-separated host list, longest-first (default n1,n2,n3,n4)",
    )
    parser.add_argument(
        "--csv",
        default="/workspace/results/m3_histogram_eq.csv",
        help="CSV output path (appended to)",
    )
    parser.add_argument(
        "--image-label",
        default="",
        help="Optional label written into the CSV (defaults to image basename)",
    )
    parser.add_argument(
        "--worker",
        default=str(Path(__file__).resolve().parent / "bench_mpi_run.py"),
        help="Path to bench_mpi_run.py (rarely changed)",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", "python"),
        help="Python executable to use inside mpirun (default 'python')",
    )
    parser.add_argument(
        "--extra-mpirun",
        default="",
        help="Extra arguments forwarded to mpirun (single string, will be split)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    ranks = _parse_int_list(args.ranks)
    if not ranks:
        parser.error("--ranks must contain at least one integer")
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    label = args.image_label or Path(args.image).name
    extra = shlex.split(args.extra_mpirun) if args.extra_mpirun else []

    records: list[dict] = []
    for world_size in ranks:
        rec = _run_one(
            image=args.image,
            world_size=world_size,
            runs=args.runs,
            warmup=args.warmup,
            hosts=hosts,
            worker_script=args.worker,
            extra_mpirun=extra,
            label=label,
            python_bin=args.python_bin,
        )
        if not rec.get("validated", False):
            sys.stderr.write(
                f"[bench] WARNING: P={world_size} produced output that disagrees "
                f"with the serial reference!\n"
            )
        records.append(rec)

    _print_table(records)
    _write_csv(records, Path(args.csv), image_label=label)
    print(f"[bench] CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
