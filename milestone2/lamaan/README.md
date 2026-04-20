# Milestone 2: Theoretical Complexity Analysis
## PRAM Model & Amdahl's Law for 4-Node Distributed FFT

**Author:** Lamaan Ali Bukhsh  
**Course:** Parallel and Distributed Computing (PDC)  
**Project:** Intelligent Edge Distributed Image Processing System  
**Milestone:** 2 (Theory & Shared Memory Algorithms)

---

## Overview

This document provides a formal theoretical analysis of the distributed 2D FFT pipeline designed for a **4-node Raspberry Pi cluster** (1 Master, 3 Workers), not the 6-node system mentioned in the original proposal.

### Key Contributions

1. **PRAM Complexity Analysis**
   - Derives Work and Span bounds for:
     - Row-FFT phase
     - Global matrix transposition
     - Column-FFT phase
     - Gaussian High-Pass Filter (GHPF)
     - Inverse 2D FFT (IFFT)
   - Identifies transposition as the primary bottleneck

2. **Amdahl's Law Numerical Analysis**
   - Estimates parallelizable fraction: $f \approx 0.736$ (73.6%)
   - Predicts speedup for 4-node system: $S(4) = 2.23\times$
   - Calculates asymptotic limit: $S_{\max} = 3.79\times$
   - Analyzes efficiency degradation with process count

3. **Communication Cost Quantification**
   - Models network delays (latency + bandwidth)
   - Measures communication-to-computation ratio: $\sim 20:1$
   - Identifies MPI collective operations as key bottleneck

4. **Optimization Recommendations**
   - Tier 1: Non-blocking MPI communication
   - Tier 2: All-to-All transpose optimization
   - Tier 3: Distributed transposition algorithms

---

## System Configuration (4-Node)

### Hardware
| Component | Specification |
|-----------|---|
| **Cluster Size** | 4 nodes (1 Master + 3 Workers) |
| **Node Hardware** | Raspberry Pi 4 (1GB RAM, ARM Cortex-A72) |
| **Network** | Gigabit Ethernet (or WiFi 802.11ac) |

### Problem Size
| Parameter | Value |
|-----------|-------|
| **Image Dimensions** | $M \times N = 1024 \times 1024$ pixels |
| **Data Type** | Complex64 (8 bytes/element) |
| **Total Input** | 8 MB |
| **Work per Worker** | $256 \times 1024$ elements |

---

## Key Results

### Complexity Bounds

| Phase | Total Work | Span | Parallelism |
|-------|-----------|------|-------------|
| Row-FFT | $MN \log N$ | $N \log N$ | $M/P$ |
| Transposition | $MN$ | $MN$ | **1** (bottleneck) |
| Column-FFT | $MN \log N$ | $N \log N$ | $M/P$ |
| **Total 2D FFT** | **$2MN \log N + MN$** | **$MN + N\log N$** | **13.95** |

For $M = N = 1024$:
- **Total Work:** $\approx 44$ million operations
- **Span (PRAM time):** $\approx 3.16$ million operations
- **Ideal speedup (∞ processors):** $\approx 13.95\times$ (theoretical maximum)
- **Practical speedup (4 nodes):** $\approx 2.23\times$ (including communication overhead)

### Amdahl's Law Analysis

**Parallelizable Fraction:** $f = 0.736$ (73.6%)  
**Sequential Fraction:** $(1-f) = 0.264$ (26.4%)

| Processors | Speedup | Efficiency |
|-----------|---------|-----------|
| 1 | 1.000x | 100% |
| 2 | 1.602x | 80.1% |
| 4 | **2.232x** | 55.8% |
| 8 | 2.666x | 33.3% |
| $\infty$ | 3.788x | - |

**Interpretation:** On our 4-node system, expect $\sim 2.2\times$ speedup. Beyond 4 nodes, communication overhead dominates.

### Communication Overhead

| Phase | Latency | Bandwidth | Total Time |
|-------|---------|-----------|-----------|
| Scatter (Master→Workers) | 10 µs | 125 MB/s | 16 ms |
| Gather (Workers→Master) | 10 µs | 125 MB/s | 16 ms |
| Transposition (on Master) | - | - | 10 ms |
| **Total per FFT cycle** | - | - | **~64 ms** |

**Computation time (PRAM, ignoring overhead):** $\sim 3.2$ ms  
**Communication-to-Computation ratio:** $64 / 3.2 = 20:1$ ⚠️

---

## Building the PDF

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- `pdflatex` compiler
- Standard packages: `amsmath`, `graphicx`, `hyperref`, `booktabs`, etc.

### Compilation

```bash
cd milestone2/lamaan/
pdflatex -interaction=nonstopmode analysis.tex
```

This generates `analysis.pdf`.

**Note:** Run `pdflatex` twice to ensure proper cross-references and table of contents:

```bash
pdflatex -interaction=nonstopmode analysis.tex
pdflatex -interaction=nonstopmode analysis.tex
```

### Output
- `analysis.pdf` — Formatted final document (~10 pages)
- `analysis.aux` — Auxiliary file (cross-references)
- `analysis.log` — Compilation log

### On Docker / Remote Systems

If building on the cluster:

```bash
docker run --rm -v $(pwd):/data texlive/texlive:latest pdflatex -output-directory=/data /data/analysis.tex
```

---

## File Structure

```
milestone2/
├── lamaan/
│   ├── analysis.tex          # Main LaTeX source (this analysis)
│   ├── preamble.sty          # LaTeX preamble & styling
│   ├── README.md             # This file
│   ├── figures/              # Directory for plots/diagrams
│   │   ├── speedup_curve.png
│   │   ├── communication_timeline.png
│   │   └── bottleneck_breakdown.png
│   └── analysis.pdf          # Compiled output (generated)
│
├── muhammad/
│   └── shared_memory_impl/   # Muhammad's IPC implementation
│
├── talha/
│   └── benchmarks/           # Talha's comparative measurements
│
└── sharjeel/
    └── validation_report/    # Sharjeel's empirical validation
```

---

## Key Insights for Team

### For Muhammad (IPC Implementation)
- Expected speedup ceiling with shared memory: $\sim 3.8\times$ (no network overhead)
- Target: Eliminate transposition bottleneck via in-process memory sharing
- Success metric: $S(4) > 3.0\times$ (beating distributed version)

### For Talha (Benchmarking)
- Measure actual performance against predictions:
  - Single-node baseline (reference)
  - 4-node distributed (compare to $2.23\times$ prediction)
  - Pure computation time vs. communication time
- Identify which phase deviates most from theory

### For Sharjeel (Validation & Reporting)
- Collect empirical data from all three configurations
- Compare measured speedup $S_{\text{measured}}(4)$ to theoretical $S_{\text{theory}}(4) = 2.23\times$
- If measured $S \approx 2.0$–$2.5\times$: theory validated ✓
- If measured $S < 2.0\times$: communication overhead higher than expected
- If measured $S > 2.5\times$: excellent optimization or estimation was conservative
- Perform sensitivity analysis: how does network latency affect speedup?

---

## Next Steps

1. **Muhammad:** Implement shared memory version; aim for $3.0–3.5\times$ speedup
2. **Talha:** Benchmark all three configurations; collect timing profiles
3. **Sharjeel:** Compare empirical results to theoretical predictions
4. **All:** Refine analysis based on findings; iterate on optimizations

---

## References

- Blelloch, G. E., Maggs, B. M., & Plaxton, C. G. (1994). "The Space-Efficient Algorithm Design Manual."
- Amdahl, G. M. (1967). "Validity of the single processor approach to achieving large scale computing capabilities." *AFIPS Conference Proceedings*, 30, 483–485.
- MPI Forum. (2021). "MPI: A Message-Passing Interface Standard (Version 4.0)."
- Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series." *Mathematics of Computation*, 19(90), 297–301.

---

## Contact

For questions or clarifications on this analysis:
- **Lamaan Ali Bukhsh** (Author)
- Email: [your-email]
- Office Hours: [schedule]

---

*Last Updated: April 20, 2026*  
*Document Version: 1.0*
