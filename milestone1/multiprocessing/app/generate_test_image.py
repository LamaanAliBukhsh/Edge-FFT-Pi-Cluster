"""
Milestone 1 – Test Image Generator
===================================

Creates a simple test image with geometric shapes for consistent benchmarking
across all implementations (Sequential, Threading, Multiprocessing).

The image is 256x256 pixels with:
  - Black background
  - White circles
  - White squares
  - Gray gradients

Perfect for testing Sobel edge detection.

Usage:
    python generate_test_image.py [--output path/to/image.jpg] [--size WxH]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_test_image(width: int = 256, height: int = 256) -> Image.Image:
    """
    Create a test image with geometric shapes for edge detection benchmarking.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image object (RGB mode)
    """
    # Create black background
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Add white circles
    margin = 30
    radius = 25
    draw.ellipse([margin, margin, margin + 2*radius, margin + 2*radius],
                 fill=(255, 255, 255))
    draw.ellipse([width - margin - 2*radius, margin,
                  width - margin, margin + 2*radius],
                 fill=(255, 255, 255))

    # Add white squares
    square_size = 40
    draw.rectangle([margin, height - margin - square_size,
                    margin + square_size, height - margin],
                   fill=(255, 255, 255))
    draw.rectangle([width - margin - square_size, height - margin - square_size,
                    width - margin, height - margin],
                   fill=(255, 255, 255))

    # Add a gradient region (edges of varying intensity)
    pixels = img.load()
    grad_start_x = width // 2 - 30
    grad_end_x = width // 2 + 30
    grad_y = height // 2
    for x in range(grad_start_x, grad_end_x):
        intensity = int(255 * (x - grad_start_x) / (grad_end_x - grad_start_x))
        for dy in range(-20, 21):
            y = grad_y + dy
            if 0 <= y < height:
                pixels[x, y] = (intensity, intensity, intensity)

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate test image for Sobel benchmarking")
    parser.add_argument("--output", default="test_image.jpg",
                        help="Output image path (default: test_image.jpg)")
    parser.add_argument("--size", default="256x256",
                        help="Image size as WIDTHxHEIGHT (default: 256x256)")
    args = parser.parse_args()

    width, height = map(int, args.size.split('x'))

    print(f"\nGenerating test image: {args.output}")
    print(f"  Size    : {width}x{height} pixels")
    print(f"  Content : Circles, squares, gradient (for edge detection)")

    img = create_test_image(width, height)
    img.save(args.output, quality=95)

    file_size = Path(args.output).stat().st_size
    print(f"  Saved   : {file_size:,} bytes\n")


if __name__ == "__main__":
    main()
