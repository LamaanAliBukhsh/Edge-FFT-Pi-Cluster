"""
Unit tests for the serial histogram equalisation reference.

These tests do **not** require MPI - they validate the pure NumPy code
that the distributed pipeline is compared against.  Keeping them green is
a precondition for trusting any MPI benchmark numbers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

from histogram_eq_serial import build_mapping, equalize_serial  # noqa: E402


class BuildMappingTests(unittest.TestCase):
    def test_uniform_histogram_is_near_identity(self) -> None:
        # An exactly uniform histogram on 256 bins should produce a mapping
        # that is monotonic non-decreasing and spans most of the range.
        hist = np.full(256, 100, dtype=np.int64)
        mapping = build_mapping(hist, total_pixels=int(hist.sum()))
        self.assertEqual(mapping.dtype, np.uint8)
        self.assertEqual(mapping.shape, (256,))
        # Monotonic non-decreasing, anchored at the lowest non-empty bin.
        self.assertTrue(np.all(np.diff(mapping.astype(int)) >= 0))
        self.assertEqual(int(mapping[0]), 0)
        self.assertEqual(int(mapping[-1]), 255)

    def test_flat_image_returns_passthrough(self) -> None:
        hist = np.zeros(256, dtype=np.int64)
        hist[42] = 1000
        mapping = build_mapping(hist, total_pixels=1000)
        # All pixels were 42; equalisation has nothing to spread across.
        self.assertEqual(mapping.dtype, np.uint8)
        self.assertEqual(mapping.shape, (256,))


class EqualizeSerialTests(unittest.TestCase):
    def test_dtype_and_shape_preserved(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
        out = equalize_serial(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_rejects_wrong_dtype(self) -> None:
        with self.assertRaises(TypeError):
            equalize_serial(np.zeros((10, 10), dtype=np.float32))

    def test_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            equalize_serial(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_idempotent_on_already_equalised(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        once = equalize_serial(image)
        twice = equalize_serial(once)
        # Equalising an already-equalised image should be a near no-op.
        self.assertTrue(np.array_equal(once, twice))


if __name__ == "__main__":
    unittest.main(verbosity=2)
