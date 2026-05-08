"""
Milestone 3 - Row-chunking helper for non-divisible image sizes
================================================================

Roadmap M3 Task 07 ("Handle Non-Divisible Image Sizes with MPI_Scatterv")
calls for a `compute_chunks(M, P)` helper that returns one row count per
worker.  When M is not divisible by P, the remainder rows are spread across
the first ``M % P`` workers so the workload is as balanced as possible.

The helper is unit-tested in ``milestone3/tests/test_chunking.py`` and is
reused both by the MPI histogram-equalisation pipeline and by the benchmark
harness when computing per-rank counts/displacements for ``Scatterv``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_chunks(M: int, P: int) -> List[int]:
    """Return ``P`` row counts that sum to ``M`` and differ by at most 1.

    The first ``M % P`` workers each receive one extra row so the
    imbalance is kept to a single row regardless of ``M`` and ``P``.

    Parameters
    ----------
    M:
        Total number of rows in the image.
    P:
        Number of MPI ranks participating in the scatter.

    Returns
    -------
    list[int]
        Row count assigned to each rank, length ``P``.
    """
    if M < 0:
        raise ValueError(f"M must be non-negative, got {M}")
    if P <= 0:
        raise ValueError(f"P must be positive, got {P}")

    base, remainder = divmod(M, P)
    counts = [base + 1 if rank < remainder else base for rank in range(P)]
    assert sum(counts) == M, "compute_chunks must preserve total row count"
    return counts


def chunks_to_scatterv(
    row_counts: List[int], width: int, dtype: np.dtype = np.uint8
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate row counts into ``(sendcounts, displacements)`` in *elements*.

    MPI's ``Scatterv``/``Gatherv`` work in element units (not bytes) when the
    buffer is described with a numpy dtype, so we convert rows to pixels by
    multiplying by ``width``.  Displacements are computed via cumulative sum.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    sendcounts = np.asarray(
        [count * width for count in row_counts], dtype=np.int64
    )
    # Element-wise prefix sum gives the offset of each rank's slice.
    displs = np.zeros_like(sendcounts)
    if sendcounts.size > 1:
        displs[1:] = np.cumsum(sendcounts[:-1])

    # numpy dtype is accepted by mpi4py through the (buf, counts, displs, type)
    # tuple form; the caller picks the actual MPI datatype.
    _ = dtype  # documentation only
    return sendcounts, displs


def row_offsets(row_counts: List[int]) -> List[int]:
    """Return the starting row index for each rank (used for assembly logging)."""
    offsets = [0]
    for count in row_counts[:-1]:
        offsets.append(offsets[-1] + count)
    return offsets


__all__ = ["compute_chunks", "chunks_to_scatterv", "row_offsets"]
