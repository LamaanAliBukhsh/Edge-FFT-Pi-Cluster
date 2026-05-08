"""
Unit tests for ``milestone3.app.chunking`` (no MPI required).

Run from inside any container with ``python -m pytest`` or directly with
``python milestone3/tests/test_chunking.py``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Make ``milestone3/app`` importable when running directly.
THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

from chunking import chunks_to_scatterv, compute_chunks, row_offsets  # noqa: E402


class ComputeChunksTests(unittest.TestCase):
    def test_divisible_case(self) -> None:
        self.assertEqual(compute_chunks(100, 4), [25, 25, 25, 25])

    def test_remainder_one_goes_to_first(self) -> None:
        self.assertEqual(compute_chunks(101, 4), [26, 25, 25, 25])

    def test_remainder_three_distributed(self) -> None:
        self.assertEqual(compute_chunks(103, 4), [26, 26, 26, 25])

    def test_more_workers_than_rows(self) -> None:
        self.assertEqual(compute_chunks(3, 4), [1, 1, 1, 0])

    def test_zero_rows(self) -> None:
        self.assertEqual(compute_chunks(0, 4), [0, 0, 0, 0])

    def test_total_matches_M_for_many_random_sizes(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            M = int(rng.integers(0, 10_000))
            P = int(rng.integers(1, 33))
            counts = compute_chunks(M, P)
            self.assertEqual(sum(counts), M)
            # Imbalance must never exceed one row.
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_chunks(-1, 4)
        with self.assertRaises(ValueError):
            compute_chunks(10, 0)


class ScattervHelpersTests(unittest.TestCase):
    def test_displacements_match_cumulative_sum(self) -> None:
        counts = compute_chunks(1025, 4)
        sendcounts, displs = chunks_to_scatterv(counts, width=1025)
        np.testing.assert_array_equal(
            sendcounts, np.array([c * 1025 for c in counts], dtype=np.int64)
        )
        # First displacement is always zero; the rest are cumulative sums.
        self.assertEqual(int(displs[0]), 0)
        np.testing.assert_array_equal(displs[1:], np.cumsum(sendcounts[:-1]))

    def test_offsets_in_rows(self) -> None:
        self.assertEqual(row_offsets([26, 25, 25, 25]), [0, 26, 51, 76])

    def test_zero_width_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunks_to_scatterv([1, 2, 3], width=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
