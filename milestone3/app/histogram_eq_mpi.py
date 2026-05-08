"""
Milestone 3 - Distributed histogram equalisation over MPI
==========================================================

Implements the canonical scatter/allreduce/gather pipeline required by the
M3 roadmap:

    Task 06  Distributed Histogram Equalization via MPI_Scatter/Gather
    Task 07  Handle Non-Divisible Image Sizes with MPI_Scatterv

The algorithm is identical in both flows; the only difference is whether the
image height is divisible by the world size:

    1. Rank 0 loads the image and decides per-rank row counts via
       ``compute_chunks``.
    2. The image is distributed with ``Scatterv`` (or ``Scatter`` if every
       rank gets the same number of rows).
    3. Each rank builds a 256-bin local histogram of its slice.
    4. ``Allreduce(SUM)`` produces the global histogram on every rank.
    5. Every rank derives the same uint8 mapping table from the global
       histogram and applies it to its local slice (no broadcast needed).
    6. ``Gatherv`` (or ``Gather``) reassembles the equalised image on rank 0,
       which optionally writes it to disk.

Run from the master container with::

    mpirun --allow-run-as-root --host n1,n2,n3,n4 -np 4 \\
        python /workspace/milestone3/app/histogram_eq_mpi.py \\
        --image /workspace/milestone3/data/lena_1024.png \\
        --output /workspace/milestone3/output/lena_1024_eq.png \\
        --validate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# When this script is launched by ``mpirun`` the working directory is /workspace
# in our compose layout, so we make the sibling modules importable explicitly.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chunking import chunks_to_scatterv, compute_chunks, row_offsets
from histogram_eq_serial import (
    build_mapping,
    equalize_serial,
    load_grayscale,
    save_grayscale,
)

from mpi4py import MPI


def _scatter_rows(
    comm: "MPI.Intracomm",
    image: np.ndarray | None,
    height: int,
    width: int,
    row_counts: list[int],
) -> np.ndarray:
    """Distribute image rows from rank 0 using Scatterv (variable counts)."""
    sendcounts, displs = chunks_to_scatterv(row_counts, width)
    local_rows = row_counts[comm.Get_rank()]
    local = np.empty((local_rows, width), dtype=np.uint8)

    sendbuf = None
    if comm.Get_rank() == 0:
        if image is None:
            raise RuntimeError("rank 0 must hold the image before scatter")
        # ``Scatterv`` needs a contiguous buffer; numpy arrays from PIL
        # already are, but ``np.ascontiguousarray`` makes this explicit.
        sendbuf = (
            np.ascontiguousarray(image),
            sendcounts,
            displs,
            MPI.UNSIGNED_CHAR,
        )

    recvbuf = (local, MPI.UNSIGNED_CHAR)
    comm.Scatterv(sendbuf, recvbuf, root=0)
    _ = height  # documented for clarity; height is implied by row_counts
    return local


def _gather_rows(
    comm: "MPI.Intracomm",
    local: np.ndarray,
    height: int,
    width: int,
    row_counts: list[int],
) -> np.ndarray | None:
    """Reassemble the image on rank 0 with Gatherv."""
    sendcounts, displs = chunks_to_scatterv(row_counts, width)

    if comm.Get_rank() == 0:
        output = np.empty((height, width), dtype=np.uint8)
        recvbuf = (output, sendcounts, displs, MPI.UNSIGNED_CHAR)
    else:
        output = None
        recvbuf = None

    comm.Gatherv(
        (np.ascontiguousarray(local), MPI.UNSIGNED_CHAR),
        recvbuf,
        root=0,
    )
    return output


def equalize_distributed(
    comm: "MPI.Intracomm",
    image: np.ndarray | None,
) -> tuple[np.ndarray | None, dict]:
    """Run histogram equalisation across ``comm`` and return ``(output, timings)``.

    On rank 0 ``image`` must be a 2D uint8 array; on every other rank the
    argument is ignored (and may be ``None``).  The returned ``output`` is
    ``None`` on non-root ranks.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ------------------------------------------------------------------
    # Step 1 - rank 0 publishes the geometry so every rank can size its
    #          local buffer before the scatter call.
    # ------------------------------------------------------------------
    if rank == 0:
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("rank 0 must provide a 2D uint8 image")
        height, width = image.shape
        shape = np.array([height, width], dtype=np.int64)
    else:
        shape = np.empty(2, dtype=np.int64)
    comm.Bcast([shape, MPI.LONG_LONG], root=0)
    height, width = int(shape[0]), int(shape[1])

    row_counts = compute_chunks(height, size)
    timings: dict[str, float] = {
        "height": height,
        "width": width,
        "world_size": size,
        "row_counts": row_counts,
        "row_offsets": row_offsets(row_counts),
    }

    # ------------------------------------------------------------------
    # Step 2 - Scatterv (works for divisible AND non-divisible heights).
    # ------------------------------------------------------------------
    comm.Barrier()
    t_scatter = MPI.Wtime()
    local = _scatter_rows(comm, image, height, width, row_counts)
    comm.Barrier()
    timings["scatter_s"] = MPI.Wtime() - t_scatter

    # ------------------------------------------------------------------
    # Step 3 - local histogram on the assigned slice.
    # ------------------------------------------------------------------
    t_hist = MPI.Wtime()
    local_hist, _ = np.histogram(local, bins=256, range=(0, 256))
    local_hist = local_hist.astype(np.int64)
    timings["local_hist_s"] = MPI.Wtime() - t_hist

    # ------------------------------------------------------------------
    # Step 4 - Allreduce so every rank ends up with the same global hist.
    # ------------------------------------------------------------------
    global_hist = np.zeros_like(local_hist)
    comm.Barrier()
    t_red = MPI.Wtime()
    comm.Allreduce(
        [local_hist, MPI.LONG_LONG],
        [global_hist, MPI.LONG_LONG],
        op=MPI.SUM,
    )
    comm.Barrier()
    timings["allreduce_s"] = MPI.Wtime() - t_red

    # ------------------------------------------------------------------
    # Step 5 - identical mapping derived independently on each rank.
    # ------------------------------------------------------------------
    t_map = MPI.Wtime()
    mapping = build_mapping(global_hist, total_pixels=height * width)
    local_eq = mapping[local]
    timings["apply_s"] = MPI.Wtime() - t_map

    # ------------------------------------------------------------------
    # Step 6 - Gatherv into the master's output buffer.
    # ------------------------------------------------------------------
    comm.Barrier()
    t_gather = MPI.Wtime()
    output = _gather_rows(comm, local_eq, height, width, row_counts)
    comm.Barrier()
    timings["gather_s"] = MPI.Wtime() - t_gather

    timings["total_compute_s"] = (
        timings["scatter_s"]
        + timings["local_hist_s"]
        + timings["allreduce_s"]
        + timings["apply_s"]
        + timings["gather_s"]
    )

    return output, timings


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def _format_row_counts(counts: list[int]) -> str:
    return ", ".join(f"r{rank}={c}" for rank, c in enumerate(counts))


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description="Distributed histogram equalisation (M3 task 6/7)"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to grayscale (or convertible) input image",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the equalised PNG (rank 0 only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare against the serial reference on rank 0 (np.array_equal)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the pipeline N times (>=1); the last result is used",
    )
    args = parser.parse_args()

    image: np.ndarray | None = None
    load_s = 0.0
    if rank == 0:
        t0 = time.perf_counter()
        image = load_grayscale(args.image)
        load_s = time.perf_counter() - t0
        print(
            f"[m3] loaded {args.image} shape={image.shape} dtype={image.dtype} "
            f"in {load_s:.4f}s",
            flush=True,
        )

    output: np.ndarray | None = None
    timings: dict = {}
    for _ in range(max(1, args.repeat)):
        output, timings = equalize_distributed(comm, image)

    if rank == 0:
        print(
            f"[m3] world_size={size} height={timings['height']} "
            f"width={timings['width']} chunks=[{_format_row_counts(timings['row_counts'])}]",
            flush=True,
        )
        print(
            "[m3] timings (s) "
            f"scatter={timings['scatter_s']:.4f} "
            f"hist={timings['local_hist_s']:.4f} "
            f"allreduce={timings['allreduce_s']:.4f} "
            f"apply={timings['apply_s']:.4f} "
            f"gather={timings['gather_s']:.4f} "
            f"total={timings['total_compute_s']:.4f}",
            flush=True,
        )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            assert output is not None
            save_grayscale(output, out_path)
            print(f"[m3] wrote equalised image to {out_path}", flush=True)

        if args.validate:
            assert image is not None and output is not None
            reference = equalize_serial(image)
            ok = np.array_equal(reference, output)
            print(f"[m3] validation vs serial reference: {'PASS' if ok else 'FAIL'}", flush=True)
            return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
