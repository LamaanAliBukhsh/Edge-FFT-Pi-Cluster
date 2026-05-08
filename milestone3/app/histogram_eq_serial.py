"""
Milestone 3 - Serial histogram equalisation reference
======================================================

This module provides the ground-truth single-process implementation against
which the distributed MPI version is validated.  It is also used by the
benchmark harness as the ``P=1`` baseline (no MPI overhead at all).

Histogram equalisation algorithm (textbook, 8-bit grayscale):
    1. Build a 256-bin histogram of pixel intensities.
    2. Compute the cumulative distribution function (CDF).
    3. Build a mapping ``v -> round(255 * (CDF[v] - CDF_min) / (N - CDF_min))``
       where ``N`` is the total pixel count.  Subtracting ``CDF_min`` keeps
       the lowest non-empty bin at 0 (the standard formulation used by
       OpenCV and scikit-image's ``equalize_hist`` reference path).
    4. Apply the mapping pixel-by-pixel.

Keeping the implementation deterministic and self-contained means the MPI
pipeline can compare its output directly with ``np.array_equal`` without
worrying about library version differences.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Tuple

import numpy as np
from PIL import Image


def build_mapping(histogram: np.ndarray, total_pixels: int) -> np.ndarray:
    """Return the 256-entry uint8 lookup table for histogram equalisation."""
    if histogram.shape != (256,):
        raise ValueError(f"histogram must have shape (256,), got {histogram.shape}")
    if total_pixels <= 0:
        raise ValueError(f"total_pixels must be positive, got {total_pixels}")

    cdf = np.cumsum(histogram).astype(np.float64)
    # First non-zero bin (CDF_min) anchors the mapping at 0.
    nonzero = cdf[cdf > 0]
    cdf_min = float(nonzero[0]) if nonzero.size else 0.0
    denom = total_pixels - cdf_min
    if denom <= 0:
        # Degenerate case: a flat image already has nothing to equalise.
        return np.arange(256, dtype=np.uint8)

    mapping = np.round((cdf - cdf_min) / denom * 255.0)
    mapping = np.clip(mapping, 0, 255).astype(np.uint8)
    return mapping


def equalize_serial(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalisation to a 2D uint8 grayscale image."""
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape {image.shape}")

    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    mapping = build_mapping(histogram, total_pixels=image.size)
    return mapping[image]


def equalize_serial_timed(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run :func:`equalize_serial` while recording wall-clock time."""
    start = time.perf_counter()
    output = equalize_serial(image)
    elapsed = time.perf_counter() - start
    return output, elapsed


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load any image file as a 2D uint8 grayscale array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_grayscale(array: np.ndarray, path: str | Path) -> None:
    """Persist a 2D uint8 array as a PNG (used for visual verification)."""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


__all__ = [
    "build_mapping",
    "equalize_serial",
    "equalize_serial_timed",
    "load_grayscale",
    "save_grayscale",
]
