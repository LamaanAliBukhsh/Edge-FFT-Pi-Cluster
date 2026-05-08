"""
Milestone 3 - Generate deterministic test images for the M3 deliverables
=========================================================================

The roadmap explicitly mentions verifying the Scatterv path on
``100x100``, ``101x100``, ``1025x1025`` (and similarly awkward) image sizes.
We synthesise the images here from a fixed RNG seed so every team member
gets byte-identical inputs, regardless of whether they have a CIFAR-10
checkout handy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# Deterministic seed - changing this invalidates all benchmark CSVs.
SEED = 0xC0FFEE

# (height, width, label) tuples covering divisible and non-divisible heights.
DEFAULT_SIZES: list[tuple[int, int, str]] = [
    (100, 100, "100x100_div4"),         # divisible by 4 (clean Scatter)
    (101, 100, "101x100_nondiv4"),      # 101 % 4 = 1   (Scatterv path)
    (103, 100, "103x100_nondiv4"),      # 103 % 4 = 3   (Scatterv path)
    (256, 256, "256x256"),
    (512, 512, "512x512"),
    (1024, 1024, "1024x1024_div4"),
    (1025, 1025, "1025x1025_nondiv4"),  # roadmap test case
    (1280, 720, "1280x720_hd"),         # COCO-style HD
]


def _make_image(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Return an 8-bit grayscale image with structured + random content.

    Pure noise has a uniform histogram and trivially equalises to itself, so
    we layer a smooth radial gradient on top.  This guarantees the
    equaliser actually changes the pixel distribution and exposes any bug
    in chunk boundaries or mapping construction.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if radial.max() > 0:
        radial = radial / radial.max()
    noise = rng.standard_normal(size=(height, width), dtype=np.float32) * 0.15
    image = np.clip(0.55 - 0.4 * radial + noise, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def generate(out_dir: Path, sizes: list[tuple[int, int, str]] | None = None) -> list[Path]:
    """Write one PNG per requested ``(height, width, label)`` triple."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths: list[Path] = []
    for height, width, label in (sizes or DEFAULT_SIZES):
        image = _make_image(height, width, rng)
        path = out_dir / f"test_{label}.png"
        Image.fromarray(image, mode="L").save(path)
        paths.append(path)
        print(f"  wrote {path}  ({height}x{width})")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory to write the synthetic PNGs into",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[gen] writing test images to {out_dir} ...")
    generate(out_dir)
    print("[gen] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
