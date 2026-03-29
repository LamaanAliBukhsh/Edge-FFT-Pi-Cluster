# Milestone 1 – Sequential Sobel Edge Detector
**Author:** Lamaan Ali Bukhsh | PDC Spring 2025

---

## Overview

This milestone implements a **sequential (non-parallel) Sobel edge detector** using pure Python loops or NumPy vectorization, but **always single-core/non-threaded**.

This baseline is critical for Milestone 1:

| Member | Task |
|---|---|
| **Lamaan Ali Bukhsh** | **Sequential Sobel baseline (this implementation)** |
| Talha Mudassar | Multi-threaded Sobel (for comparison) |
| Muhammad Fahad | Multiprocessing pool version (for GIL escape proof) |
| Sharjeel Chandna | Unified GIL analysis report |

---

## Why Sequential Matters

The **sequential implementation is the reference baseline** for all speedup calculations:

$$\text{Speedup} = \frac{\text{Time}_{\text{sequential}}}{\text{Time}_{\text{N threads/processes}}}$$

Without this baseline:
- Talha's 4-thread speedup of 4.043x has no meaning
- We cannot quantify GIL overhead
- We cannot compare threading vs. multiprocessing

---

## Algorithm Design

### Sobel Edge Detection

The Sobel operator convolves the image with two 3×3 kernels:

```
Gx = [[-1, 0, 1],    Gy = [[-1, -2, -1],
      [-2, 0, 2],          [ 0,  0,  0],
      [-1, 0, 1]]          [ 1,  2,  1]]
```

Edge magnitude at each pixel:
$$\text{magnitude}(x, y) = \sqrt{G_x^2 + G_y^2}$$

### Sequential Strategy

**No parallel decomposition.** Just load image → apply Sobel → save output.

Two implementations provided:

1. **Vectorized** (default): Uses `scipy.ndimage.convolve` for fast NumPy-based convolution
   - Single-core, but optimized by NumPy
   - Expected time: ~0.39-0.45s for 512×512
   
2. **Pure-Python** (`--naive`): Explicit nested loops
   - Single-core, no NumPy optimization
   - Expected time: ~15-20s for 512×512 (shows why vectorization matters)
   - Useful for demonstrating algorithmic slowness vs. GIL effects

---

## File Structure

```
sequential/
├── app/
│   ├── sobel_sequential.py       # Core sequential Sobel implementation
│   ├── benchmark.py              # Benchmarking harness (multiple runs)
│   └── generate_test_image.py    # Generates a synthetic test image
├── Dockerfile                    # Single-node container
├── run.ps1                       # PowerShell launch script
└── README.md                     # This file
```

---

## Usage

All commands are run from the `sequential/` directory.

### Quick Start: Run edge detection

```powershell
# Default: 512×512 test image, output saved to ./output/edges_sequential.png
.\run.ps1 -Command run

# Pure-Python naive mode (slower, better shows baseline)
.\run.ps1 -Command run -Naive
```

### Benchmark: Get baseline timing

```powershell
# Run Sequential Sobel 3 times, average the results
.\run.ps1 -Command benchmark

# Naive mode benchmark (64×64 to complete faster under QEMU)
.\run.ps1 -Command benchmark -Naive
```

### Interactive shell: Explore container

```powershell
.\run.ps1 -Command shell
# Inside container:
python /app/sobel_sequential.py --image /app/test_image.jpg --output /tmp/edges.png
```

---

## Expected Benchmark Results

### Vectorized (NumPy, default)

```
==============================================================================
  Milestone 1 – Sequential Sobel Benchmark
  Lamaan Ali Bukhsh | PDC Project
==============================================================================

  Image   : /app/test_image.jpg (512×512)
  Runs    : 3
  Mode    : Vectorized (NumPy)

  Run 1/3: 0.3892s
  Run 2/3: 0.3954s
  Run 3/3: 0.3895s

  Average: 0.3913s

==============================================================================
  BASELINE RESULT – Sequential Sobel
==============================================================================
  Average Time: 0.3913s
  Speedup:      1.000x  (this is the reference baseline)
  Efficiency:   100.0%  (N/A for sequential)
==============================================================================

  NOTE: All speedup calculations in M1 use this baseline.
        Threaded speedup = 0.3913s / threaded_time
```

### Speedup Comparison (vs. other M1 implementations)

| Implementation | Workers | Avg Time | Speedup | Analysis |
|---|---|---|---|---|
| **Sequential** (this) | 1 | 0.3913s | 1.000x | **BASELINE** |
| Threading (Talha) | 1 | 0.3921s | 1.000x | Single thread |
| Threading (Talha) | 4 | 0.0970s | 4.043x | GIL + NumPy limits |
| Multiprocessing (Fahad) | 1 | 0.3921s | 1.000x | Process overhead |
| Multiprocessing (Fahad) | 4 | ~0.1000s | ~3.91x | True parallelism (expected) |

---

## How This Feeds Into M1 Report

Sharjeel Chandna will use this baseline to create the unified M1 analysis:

1. **GIL Impact Quantified**: Threading speedup vs. Sequential speedup
   - Threading should be < 4x for 4 threads (GIL contention)
   - Multiprocessing should be closer to 4x (no GIL)

2. **Efficiency Curve**: How well each approach scales
   - Sequential: flat (always 1x)
   - Threading: drops with thread count due to GIL
   - Multiprocessing: more linear due to true parallelism

3. **Recommendations for M2/M3**
   - Shared memory parallelism on single Pi: multiprocessing wins
   - Distributed across 4-node cluster: MPI (M3) will need careful design

---

## Technical Notes

### Why NumPy Convolution is Fast

Even though sequential, NumPy's `convolve()` is implemented in C and SIMD-optimized:
- Vectorized operations operate on arrays, not single pixels
- CPU cache locality improves
- BLAS libraries (if installed) provide matrix operation speedups

Sequential NumPy ≠ slow. It just means single-core, not parallelized.

### Replicate Padding

Image boundaries are handled via **replicate padding** (clamp pixel value to nearest valid pixel):
- `Sobel(pixel)` at image edge uses replicated boundary pixels
- Avoids artifacts from zero-padding or circular wrapping

### Image Normalization

Output edge magnitude is normalized to [0, 255] for visualization:
$$\text{output}[x, y] = \frac{\text{magnitude}[x, y]}{\max(\text{magnitude})} \times 255$$

This ensures the edge image uses full dynamic range.

---

## Integration with Milestone 1

This baseline:
1. ✅ Provides reference speedup denominator
2. ✅ Demonstrates what "single-core" looks like
3. ✅ Feeds into Sharjeel's GIL analysis
4. ✅ Prepares for M2 (shared memory) and M3 (distributed)

**Keep this implementation stable.** Changes to the Sobel kernel or image loading must be reflected in Talha's threading and Fahad's multiprocessing versions for fair comparison.

---

## Future Work

- **Milestone 2**: Extend sequential to use shared memory IPC (single node, multiple processes)
- **Milestone 3**: Implement distributed 2D FFT using MPI across 4-node cluster
- **Milestone 4**: Optimize communication/computation overlap with non-blocking MPI

---

**Status:** ❌ Awaiting Talha's threading results and Fahad's multiprocessing for comparison.
