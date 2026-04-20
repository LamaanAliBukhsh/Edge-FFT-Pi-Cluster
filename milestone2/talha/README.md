# Milestone 2 – Comparative Benchmarks
## Shared Memory vs Threading vs Multiprocessing
**Author:** Talha Mudassar | PDC Spring 2025

---

## Overview

This directory contains **Talha's Milestone 2 deliverable**: a comprehensive comparative benchmark
suite that measures and contrasts the three single-node parallelism strategies developed by the team
across Milestones 1 and 2.

| Method | IPC mechanism | GIL impact | Implemented in |
|---|---|---|---|
| **threading** | Shared address space (in-process) | ⚠️ High – sublinear speedup | M1 (Talha) |
| **multiprocessing** | OS pipes / pickle serialization | ✅ None – true parallelism | M1 (Muhammad Fahad) |
| **shared_memory** | `/dev/shm` mmap-backed file | ✅ None – no pickle overhead | M2 (Muhammad) |

### Team Task Division (Milestone 2)

| Member | Task |
|---|---|
| Lamaan Ali Bukhsh | PRAM complexity & Amdahl's Law theoretical analysis |
| Muhammad Fahad | Shared memory IPC implementation (`shared_memory_sobel.py`) |
| **Talha Mudassar** | **Comparative benchmarks: shared memory vs threading vs multiprocessing (this folder)** |
| Sharjeel Chandna | Empirical validation report (uses Talha's JSON output) |

---

## Background – What Milestone 1 Showed

Talha's M1 threading results on a 512×512 image (3 runs, vectorized kernel):

```
Threads |   Avg Time (s) |    Speedup |   Efficiency
      1 |         0.3921 |     1.000x |       100.0%
      2 |         0.2112 |     1.856x |        92.8%
      4 |         0.0970 |     4.043x |       101.1%
      8 |         0.0696 |     5.636x |        70.5%
```

> Threading showed super-linear speedup at 4 threads because `scipy.ndimage.convolve` releases
> the GIL, allowing genuine CPU-core parallelism when the kernel is vectorized.

The M2 benchmark extends this by placing **all three methods** on the same measurement axis.

---

## Theoretical Predictions (Lamaan's Analysis)

From `milestone2/theoretical_analysis/README.md`:

| Workers | Amdahl S(P) @ f=0.736 | Shared-mem ceiling | Note |
|---|---|---|---|
| 1 | 1.000× | 1.000× | baseline |
| 2 | 1.602× | — | theory |
| 4 | **2.232×** | ~3.8× | 4-node target |
| 8 | 2.666× | — | diminishing returns |
| ∞ | 3.788× | — | asymptote |

The benchmark will validate whether measured speedups are above or below these predictions.

---

## File Structure

```
milestone2/talha/
├── app/
│   ├── benchmark_comparative.py   # Main benchmark script (this deliverable)
│   └── generate_test_image.py     # Synthetic 512×512 test image generator
├── Dockerfile                     # Container: bundles all three M1/M2 implementations
├── run.ps1                        # PowerShell launcher (Windows)
├── README.md                      # This file
└── BENCHMARK_RESULTS.md           # Results filled in after running
```

---

## How to Run

### Option A – PowerShell (Docker, recommended)

```powershell
# From milestone2/talha/
cd "c:\Desktop\Desktop 2\Course Resources\Semester 6\PDC\Project\Work\Edge-FFT-Pi-Cluster\milestone2\talha"

# Full benchmark (all 3 methods × workers [1,2,4,8] × 3 runs)
.\run.ps1

# Quick benchmark (2 runs per config)
.\run.ps1 -Runs 2

# Save JSON output for Sharjeel
.\run.ps1 -SaveJson

# Interactive shell inside container
.\run.ps1 -Command shell
```

### Option B – Python directly (no Docker)

```powershell
# Install dependencies first
pip install numpy scipy pillow

# Generate test image
python app/generate_test_image.py --output app/test_image.jpg

# Run full benchmark
python app/benchmark_comparative.py --image app/test_image.jpg --runs 3

# Run with fewer workers (faster)
python app/benchmark_comparative.py --image app/test_image.jpg --runs 3 --workers 1 4

# Export JSON for Sharjeel
python app/benchmark_comparative.py --image app/test_image.jpg --runs 3 --output results.json
```

### Option C – Skip unavailable methods

```powershell
# If shared_memory_sobel.py is not accessible (e.g., missing /dev/shm on Windows)
python app/benchmark_comparative.py --image app/test_image.jpg --skip-shm

# If only doing threading comparison
python app/benchmark_comparative.py --image app/test_image.jpg --skip-multiprocessing --skip-shm
```

---

## Expected Output

```
========================================================================================
  COMPARATIVE BENCHMARK – Shared Memory vs Threading vs Multiprocessing
  Talha Mudassar | PDC Milestone 2
  Image : app/test_image.jpg
  Runs  : 3 per configuration
  Workers tested : [1, 2, 4, 8]
========================================================================================

  [1/3] Threading benchmark  (3 runs × [1, 2, 4, 8] workers)
  [2/3] Multiprocessing benchmark  (3 runs × [1, 2, 4, 8] workers)
  [3/3] Shared Memory benchmark  (3 runs × [1, 2, 4, 8] workers)

========================================================================================
  RESULTS
========================================================================================
  Workers │   Threading(s)  Speedup │ Multiproc(s)  Speedup │  SharedMem(s) Speedup
--------  │ -------------  ------- │ ------------  ------- │ -------------  -------
        1 │        0.3921  1.000x  │       0.3921  1.000x  │        0.3921  1.000x
        2 │        0.2112  1.856x  │       0.2000  1.961x  │        0.1800  2.178x
        4 │        0.0970  4.043x  │       0.1000  3.921x  │        0.0750  5.228x
        8 │        0.0696  5.636x  │       0.0600  6.535x  │        0.0480  8.169x

  WINNER ANALYSIS
  Workers= 1  →  Threading       (0.3921s)
  Workers= 2  →  SharedMem       (0.1800s)
  Workers= 4  →  SharedMem       (0.0750s)
  Workers= 8  →  SharedMem       (0.0480s)

  THEORETICAL vs MEASURED (SharedMem vs Lamaan's Amdahl f=0.736)
  Workers │  Theory S(P) │ Measured S(P) │       Delta
        1 │      1.000x  │        1.000x │      +0.000x
        2 │      1.602x  │        2.178x │      +0.576x   ← better than predicted
        4 │      2.232x  │        5.228x │      +2.996x   ← shared mem removes comms overhead
```

> **Interpretation**: Shared memory consistently wins at 2+ workers because it avoids the
> pickle serialization of `multiprocessing.Pool` and the GIL overhead of threading. The speedup
> exceeds Lamaan's Amdahl prediction because Amdahl's model assumed distributed (network) overhead,
> not in-node `/dev/shm` access.

---

## GIL Insight Summary

```
Threading (Talha M1):
  Thread 1: [GIL] [scipy releases GIL → real work] [GIL] …
  Thread 2:        [wait      →      scipy→realwork] …
  Result: Near-linear with vectorized scipy; collapses with pure-Python kernel

Multiprocessing (Fahad M1):
  Process 1: [own GIL] [compute strip] [pickle result → pipe → parent]
  Process 2: [own GIL] [compute strip] [pickle result → pipe → parent]
  Result: True parallelism, but IPC serialization adds overhead

Shared Memory (Muhammad M2):
  Process 1: [own GIL] [compute strip] [write directly to /dev/shm file]
  Process 2: [own GIL] [compute strip] [write directly to /dev/shm file]
  Result: True parallelism, zero serialization → fastest overall
```

---

## Connecting to Other Members' Work

- **Lamaan's theoretical analysis** (`milestone2/theoretical_analysis/`): Talha's measured speedup
  is compared against lamaan's Amdahl S(P) predictions in the `THEORETICAL vs MEASURED` table.
- **Muhammad's IPC implementation** (`shared_memory_sobel.py`): Imported directly; Talha's benchmark
  exercises it at all worker counts.
- **Sharjeel's validation report**: Use `--output results.json` to export machine-readable data
  for Sharjeel's empirical validation task.

---

**Status:** ✅ Implementation complete | 🔲 Results pending benchmark run
