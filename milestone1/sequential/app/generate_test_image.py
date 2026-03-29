"""
Generate a synthetic test image for Milestone 1 benchmarking.
Creates grayscale images with geometric shapes (good for edge detection testing).

Usage:
    python generate_test_image.py [--size N] [--output <path>]
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw


def generate_test_image(size: int = 512, output_path: str = "/app/test_image.jpg") -> None:
    """Create a synthetic grayscale image with clear geometric edges."""
    img = Image.new("L", (size, size), color=30)
    draw = ImageDraw.Draw(img)

    # Draw geometric shapes with strong intensity contrast → clear edges
    draw.rectangle([size//8, size//8, 3*size//8, 3*size//8], fill=200)
    draw.ellipse([size//2, size//8, 7*size//8, 3*size//8], fill=180)
    draw.polygon([
        (size//2, 5*size//8),
        (3*size//8, 7*size//8),
        (5*size//8, 7*size//8),
    ], fill=220)
    draw.rectangle([size//8, size//2, 3*size//8, 7*size//8], fill=150)
    draw.line([(0, size//2), (size, size//2)], fill=240, width=3)
    draw.line([(size//2, 0), (size//2, size)], fill=240, width=3)

    # Add subtle noise to make it more realistic
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    result = Image.fromarray(arr, mode="L")
    result.save(output_path)
    print(f"[OK] Test image ({size}x{size}) saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test image")
    parser.add_argument("--size",   type=int, default=512)
    parser.add_argument("--output", default="/app/test_image.jpg")
    args = parser.parse_args()
    generate_test_image(args.size, args.output)
