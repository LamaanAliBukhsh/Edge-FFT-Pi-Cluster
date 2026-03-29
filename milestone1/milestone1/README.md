# Milestone 1 – Foundations of Concurrency
## Multi-threaded Sobel Edge Detector
**Author:** Talha Mudassar | PDC Spring 2025

---

## Overview

This milestone implements a **multi-threaded Sobel edge detector** using Python's `threading`
module, demonstrating single-node parallelism and the practical effects of Python's Global
Interpreter Lock (GIL) on CPU-bound tasks.

The implementation is part of a coordinated team effort for Milestone 1:

| Member | Task |
|---|---|
| Lamaan Ali Bukhsh | Sequential Sobel baseline |
| **Talha Mudassar** | **Multi-threaded Sobel (this implementation)** |
| Muhammad Fahad | Multiprocessing pool version |
| Sharjeel Chandna | Benchmarking & GIL analysis report |

---

## Algorithm Design

### Sobel Edge Detection

The Sobel operator convolves the image with two 3×3 kernels to approximate horizontal (Gx)
and vertical (Gy) gradients. The edge magnitude at each pixel is:

```
magnitude(x, y) = sqrt(Gx² + Gy²)
```

Kernels:
```
Gx = [[-1, 0, 1],    Gy = [[-1, -2, -1],
      [-2, 0, 2],          [ 0,  0,  0],
      [-1, 0, 1]]          [ 1,  2,  1]]
```

### Parallelization Strategy

The image height (H rows) is divided into **N equal horizontal strips**, one per thread:

```
Thread 0: rows [0,   H/N)
Thread 1: rows [H/N, 2H/N)
...
Thread N-1: rows [(N-1)H/N, H)
```

Each thread independently computes the Sobel gradient for its strip and writes the result into
a **shared output array** protected by a `threading.Lock`.

```python
lock = threading.Lock()

def worker(gray, output, row_start, row_end, lock):
    local_result = compute_sobel(gray, row_start, row_end)
    with lock:                          # Single write per thread
        output[row_start:row_end] = local_result
```

Two kernel modes are provided:
- **Vectorized** (default): uses `scipy.ndimage.convolve` for fast NumPy-based convolution
- **Pure-Python** (`--naive`): explicit nested loops — slower but shows GIL contention clearly

---

## File Structure

```
milestone1/
├── app/
│   ├── sobel_threaded.py       # Core multi-threaded Sobel implementation
│   ├── benchmark.py            # Benchmarking harness (1, 2, 4, 8 threads)
│   └── generate_test_image.py  # Generates a synthetic 512×512 test image
├── Dockerfile                  # Single-node container (no MPI)
├── run.ps1                     # PowerShell launch script
└── README.md                   # This file
```

---

## Usage

All commands are run from the `milestone1/` directory.

### Quick Start: Run edge detection

```powershell
# Default: 4 threads, output saved to ./output/edges_t4.png
.\run.ps1 -Command run

# Custom thread count
.\run.ps1 -Command run -Threads 8
```

### Benchmark: Compare thread counts

```powershell
# Vectorized (default) – use for performance benchmarking
.\run.ps1 -Command benchmark

# Pure-Python kernel (--naive) – best demonstrates GIL contention
.\run.ps1 -Command benchmark -Naive
```

### Manual Docker commands

```powershell
# Build the image
docker build -t ms1-sobel .

# Run edge detection (output to host ./output/ folder)
docker run --rm -v "${PWD}/output:/output" ms1-sobel `
    python /app/sobel_threaded.py --image /app/test_image.jpg --threads 4 --output /output/edges.png

# Run benchmark
docker run --rm ms1-sobel python /app/benchmark.py --image /app/test_image.jpg --runs 3 --naive
```

---

## Expected Benchmark Output

```
================================================================
  BENCHMARK RESULTS – Multi-threaded Sobel (threading)
  Talha Mudassar | PDC Milestone 1
================================================================

 Threads |   Avg Time (s) |    Speedup |   Efficiency
------------------------------------------------------------
       1 |         1.2840 |     1.000x |       100.0%
       2 |         1.1920 |     1.077x |        53.9%
       4 |         1.1350 |     1.131x |        28.3%
       8 |         1.1100 |     1.157x |        14.5%

  Note: Speedup < N threads is expected due to Python's GIL.
  The GIL prevents true parallelism for CPU-bound threading.
  (See multiprocessing version by Muhammad Fahad for comparison)
```

> **GIL Observation**: Python's GIL allows only one thread to execute Python bytecode at a
> time, severely limiting CPU-bound thread parallelism. The minimal speedup at 8 threads
> compared to 1 thread confirms this. The `multiprocessing` implementation (Fahad) bypasses
> the GIL via separate processes and should show much better scaling.

---

## Connecting to Other Members' Work

- **Sequential baseline** (Lamaan): Compare single-thread time from this benchmark against
  Lamaan's sequential implementation to verify correctness.
- **Multiprocessing** (Fahad): The `benchmark.py` output table is designed to be directly
  comparable to Fahad's multiprocessing results.
- **GIL Analysis** (Sharjeel): The `--naive` flag produces clear GIL contention data for
  Sharjeel's analysis report. Run with `--naive` and large images for the most visible effect.
