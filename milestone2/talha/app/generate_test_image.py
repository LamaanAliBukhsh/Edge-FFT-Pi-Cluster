"""
Generate a synthetic test image for benchmarking.
Talha Mudassar | PDC Milestone 2

Produces a 512x512 grayscale image with geometric shapes and gradients –
the same style used by teammates in M1 (Lamaan, Fahad) for fair comparison.

Usage:
    python generate_test_image.py [--output <path>] [--size WxH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_test_image(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Create a synthetic grayscale image with:
      - Radial gradient background (exercises all pixel values)
      - Concentric circles and rectangles (clear edges for Sobel)
      - Diagonal stripe pattern (mixed-frequency content)
    """
    img = np.zeros((height, width), dtype=np.float32)

    # --- Radial gradient background ---
    cx, cy = width // 2, height // 2
    y_idx, x_idx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    img += (dist / max_dist) * 128.0

    # --- Concentric rectangles (sharp edges) ---
    pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil_img)
    for shrink in range(20, min(cx, cy) - 10, 40):
        brightness = 200 if (shrink // 40) % 2 == 0 else 80
        draw.rectangle(
            [shrink, shrink, width - shrink, height - shrink],
            outline=brightness,
            width=3,
        )
    img = np.array(pil_img, dtype=np.float32)

    # --- Diagonal stripe overlay ---
    stripe_period = 32
    stripes = ((x_idx + y_idx) % stripe_period) < (stripe_period // 2)
    img += stripes.astype(np.float32) * 30.0

    # --- Three filled circles (strong circular edges) ---
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil_img)
    circles = [
        (width // 4,     height // 4,     60, 220),
        (3 * width // 4, height // 4,     50, 180),
        (width // 2,     3 * height // 4, 70, 200),
    ]
    for x0, y0, r, fill in circles:
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=fill)

    return np.array(pil_img, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test image for M2 comparative benchmark"
    )
    parser.add_argument(
        "--output",
        default="test_image.jpg",
        help="Output path for the test image (default: test_image.jpg)",
    )
    parser.add_argument(
        "--size",
        default="512x512",
        help="Image dimensions as WxH (default: 512x512)",
    )
    args = parser.parse_args()

    try:
        w_str, h_str = args.size.lower().split("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        print(f"[ERROR] Invalid --size format '{args.size}'. Use WxH, e.g. 512x512")
        sys.exit(1)

    image = generate_test_image(width, height)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(out_path)
    print(f"[OK] Test image saved -> {out_path}  ({width}x{height} px, grayscale)")


if __name__ == "__main__":
    main()
