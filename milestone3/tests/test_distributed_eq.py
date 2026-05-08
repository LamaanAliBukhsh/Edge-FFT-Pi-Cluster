"""
End-to-end MPI test: run the distributed equaliser on several image sizes
and compare the gathered output against the serial reference.

Designed to be executed inside the cluster, e.g.::

    mpirun --allow-run-as-root --host n1,n2,n3,n4 -np 4 \\
        python /workspace/milestone3/tests/test_distributed_eq.py

The script returns a non-zero exit code on any mismatch so it can be wired
into ``scripts/m3-run-tests.ps1``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

from histogram_eq_mpi import equalize_distributed  # noqa: E402
from histogram_eq_serial import equalize_serial  # noqa: E402

from mpi4py import MPI  # noqa: E402


# Image sizes covering both divisible and non-divisible heights.  At
# np=4 the non-divisible cases exercise the Scatterv code path.
TEST_SHAPES: list[tuple[int, int]] = [
    (100, 100),
    (101, 100),
    (103, 100),
    (256, 256),
    (1024, 1024),
    (1025, 1025),
    (720, 1280),
]


def _make_image(height: int, width: int, seed: int) -> np.ndarray:
    """Mirror the test image generator from ``generate_test_images.py``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if radial.max() > 0:
        radial = radial / radial.max()
    noise = rng.standard_normal(size=(height, width), dtype=np.float32) * 0.15
    image = np.clip(0.55 - 0.4 * radial + noise, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    failures = 0
    for shape in TEST_SHAPES:
        height, width = shape
        if rank == 0:
            image = _make_image(height, width, seed=1234 + height * width)
            reference = equalize_serial(image)
        else:
            image = None
            reference = None

        output, timings = equalize_distributed(comm, image)

        if rank == 0:
            assert output is not None and reference is not None
            ok = np.array_equal(output, reference)
            tag = "PASS" if ok else "FAIL"
            print(
                f"[m3-test] {tag} size={height}x{width} world_size={size} "
                f"row_counts={timings['row_counts']} "
                f"total={timings['total_compute_s']:.4f}s",
                flush=True,
            )
            if not ok:
                failures += 1
                # Help debugging: report the maximum absolute difference and
                # whether the histograms agree, so a chunking bug vs a
                # mapping bug is distinguishable.
                diff = np.abs(output.astype(int) - reference.astype(int))
                print(
                    f"          max_abs_diff={diff.max()} mismatched_pixels={int((diff>0).sum())}",
                    flush=True,
                )

    failures = comm.bcast(failures, root=0)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
