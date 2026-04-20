# Milestone 2: Theoretical Analysis Summary
## Quick Reference for Team

**Author:** Lamaan Ali Bukhsh  
**System:** 4-Node Raspberry Pi Cluster (1 Master, 3 Workers)  
**Date:** April 20, 2026

---

## Executive Summary

This document summarizes the key theoretical findings that will guide implementation and benchmarking efforts.

### **Core Prediction: 4-Node System Achieves ~2.2x Speedup**

| Metric | Value | Notes |
|--------|-------|-------|
| **Predicted Speedup** | $S(4) = 2.23x$ | Compared to single-node baseline |
| **Parallelizable Fraction** | $f = 73.6\%$ | 26.4% is sequential bottleneck |
| **Theoretical Maximum** | $S_{max} = 3.79x$ | Asymptotic limit (even with infinite processors) |
| **Expected Efficiency** | $E(4) = 55.8\%$ | Moderate; communication overhead ~44% of time |

---

## Complexity Bounds (PRAM Model)

### 2D FFT with $M = N = 1024$

| Phase | Work | Span | Notes |
|-------|------|------|-------|
| **Row-FFT** | $10.5M$ ops | $10K$ ops | Parallel across 4 workers |
| **Transposition** | $1.0M$ ops | $1.0M$ ops | **BOTTLENECK** (fully sequential) |
| **Column-FFT** | $10.5M$ ops | $10K$ ops | Parallel across 4 workers |
| **GHPF Filter** | $1.0M$ ops | $0.25M$ ops | Trivially parallel |
| **Inverse FFT** | $22.0M$ ops | $1.1M$ ops | Mirrors forward FFT |
| **Total** | **44M ops** | **3.2M ops** | Ideal PRAM: ~3.2ms computation |

**Key:** Work/Span = 13.95, meaning theoretical speedup on infinite processors is 13.95x (but limited to 3.79x by sequential fraction).

---

## Communication Analysis

### Network Parameters (Gigabit Ethernet)

| Parameter | Value |
|-----------|-------|
| Latency ($\alpha$) | 10 $\mu$s |
| Bandwidth ($\beta$) | 125 MB/s |
| Data per Worker | 2 MB (image block) |

### Communication Times

| Operation | Time |
|-----------|------|
| Single Scatter (Master to Worker) | 16 ms |
| Single Gather (Worker to Master) | 16 ms |
| Full FFT cycle (4 scatter + 4 gather + transpose) | **~64 ms** |

### Communication-to-Computation Ratio

$$\frac{T_{\text{comm}}}{T_{\text{compute}}} = \frac{64 \text{ ms}}{3.2 \text{ ms}} = 20$$

**Interpretation:** Communication is 20x longer than theoretical computation. This is the main optimization target.

---

## Amdahl's Law Predictions

### Sequential Fractions Breakdown

| Component | Time | % of Total |
|-----------|------|-----------|
| I/O (image load) | 5 ms | 1.6% |
| MPI Scatter (Row blocks) | 16 ms | 5.1% |
| MPI Gather (Row-FFT results) | 16 ms | 5.1% |
| Transposition (on master) | 10 ms | 3.2% |
| MPI Scatter (Transposed) | 16 ms | 5.1% |
| MPI Gather (Final FFT) | 16 ms | 5.1% |
| I/O + sync overhead | 16 ms | 5.1% |
| **Sequential Total** | **79 ms** | **26.4%** |
| **Parallel Computation** | **220 ms** | **73.6%** |
| **Total Execution** | **299 ms** | **100%** |

---

## Speedup vs. Processor Count

### Amdahl's Law with f = 0.736

$$S(P) = \frac{1}{0.264 + \frac{0.736}{P}}$$

| P (Nodes) | S(P) | E(P) | Loss vs. Ideal |
|-----------|------|------|---|
| 1 | 1.00x | 100% | 0% |
| 2 | 1.60x | 80% | 20% |
| 4 | **2.23x** | **56%** | **44%** |
| 6 | 2.54x | 42% | 58% |
| 8 | 2.67x | 33% | 67% |
| infinite | 3.79x | - | - |

**Insight:** 4 nodes is near the practical optimum. Beyond 6 nodes, efficiency drops sharply.

---

## Bottleneck Ranking

### Primary: Global Transposition (13.3% of total time)

**Why:** Must gather all data to master, transpose in-memory, redistribute.

**Mitigation:**
1. Use MPI_Alltoall for simultaneous exchange
2. Distribute transposition logic (workers transpose local blocks first)
3. Overlap communication and computation

**Expected Benefit:** 2-3 ms savings - S(4) increases from 2.23x to 2.35x

### Secondary: MPI Collective Operations Latency (12% of total time)

**Why:** Multiple scatter/gather phases, each incurs startup latency.

**Mitigation:**
1. Batch operations where possible
2. Use non-blocking MPI_Isend/MPI_Irecv
3. Pipeline overlapping sends/receives

**Expected Benefit:** 4-5 ms savings - S(4) increases from 2.35x to 2.55x

### Tertiary: Master Node Congestion (N/A for 4 nodes, critical for 6+)

**Why:** All-to-one communication bottleneck.

**Mitigation:** Use hypercube or tree topology communication patterns.

**Expected Benefit:** Negligible for 4 nodes; essential for scaling beyond 6.

---

## Optimization Roadmap

### Phase 1: Non-Blocking Communication (Impact: +0.3-0.5x speedup)

**Action Items:**
- Replace all MPI_Send/MPI_Recv with MPI_Isend/MPI_Irecv
- Overlap computation and communication
- Use MPI_Waitall at synchronization points

**Target:** Reduce sequential fraction from 26.4% to 15%  
**Expected S(4):** 2.23 to 2.8x

### Phase 2: All-to-All Transpose (Impact: +0.1-0.2x speedup)

**Action Items:**
- Profile MPI_Alltoall performance
- Implement optimized local transpose

**Target:** 20% faster transposition phase  
**Expected S(4):** 2.8 to 2.95x

### Phase 3: Distributed Transposition (Impact: +0.1× speedup, scales better for 6+ nodes)

**Action Items:**
- Implement peer-to-peer data exchange
- Workers coordinate transpose locally

**Target:** Eliminate central master bottleneck  
**Expected S(4):** 2.95 to 3.0x
**Expected S(6):** 2.54 to 3.2x (much better scaling)

---

## Validation Checklist for Sharjeel

Use this to validate predictions against empirical measurements:

- [ ] **Single-node baseline:** Measure time for 2D FFT on master alone
- [ ] **4-node execution:** Run full distributed pipeline, measure total time
- [ ] **Speedup calculation:** $S(4) = T_{\text{single}} / T_{\text{4node}}$
  - Expected: 2.0 <= S(4) <= 2.5x
  - If $S < 2.0$: Communication overhead worse than estimated
  - If $S > 2.5$: Excellent optimization or conservative estimate
- [ ] **Per-phase timing:**
  - Row-FFT time (should be ~50 ms on 4 workers)
  - Transposition time (should be ~10 ms on master)
  - Column-FFT time (should be ~50 ms on 4 workers)
  - Communication breakdown (should be ~60+ ms total)
- [ ] **Communication profiling:** Use MPI_Wtime to measure send/receive times
- [ ] **Efficiency calculation:** $E(4) = S(4) / 4$
  - Expected: 0.5 <= E(4) <= 0.7
- [ ] **Sensitivity:** Vary image size (256x256, 512x512, 1024x1024) and measure scaling

---

## For Muhammad (IPC Implementation)

**Your Goals:**

1. **Target Speedup:** S(4) > 3.0x (beat distributed version)
2. **Why Achievable:** Eliminate network latency + transposition overhead
3. **Key Metric:** Parallelizable fraction should approach f ~0.95+ (vs. 0.736 for distributed)

### Expected Performance

| Approach | f | S(4) | E(4) |
|----------|---|------|------|
| Distributed (Network) | 0.736 | 2.23x | 56% |
| Shared Memory (IPC) | 0.92 | 3.33x | 83% |
| Threading (M1 baseline) | 0.95 | 3.85x | 96% |

**Success Criteria:** Shared memory speedup > 3.0x indicates IPC eliminated transposition bottleneck.

---

## For Talha (Benchmarking Suite)

**Your Goals:**

1. **Measure actual speedup** vs. theoretical 2.23x
2. **Identify bottlenecks** via per-phase timing
3. **Compare three approaches:** distributed (network) vs. shared memory (IPC) vs. threading (from M1)
4. **Stress test:** Vary image size, network latency (WiFi vs. Ethernet)

### Benchmark Matrix

| Test | Setup | Expected Result |
|------|-------|---|
| Single-node baseline | 2D FFT on master | 300 ms |
| 4-node distributed | Full distributed pipeline | 135 ms (2.23x speedup) |
| 4-node with optimization | Non-blocking MPI + all-to-all | 105 ms (2.9x speedup) |
| Shared memory (Muhammad) | IPC + no network | 90 ms (3.3x speedup) |

---

## Mathematical Reference

### Amdahl's Law Formula

$$S(P) = \frac{1}{(1-f) + \frac{f}{P}}$$

where:
- $S(P)$ = speedup on $P$ processors
- $f$ = parallelizable fraction (0 to 1)
- $(1-f)$ = sequential fraction
- As $P \to \infty$: $S_{\max} = \frac{1}{1-f}$

### Our Values

$$S(4) = \frac{1}{0.264 + \frac{0.736}{4}} = \frac{1}{0.448} = 2.232$$

$$S_{\max} = \frac{1}{0.264} = 3.788$$

### Efficiency

$$E(P) = \frac{S(P)}{P}$$

$$E(4) = \frac{2.232}{4} = 0.558 = 55.8\%$$

---

## File Locations

| File | Purpose | Access |
|------|---------|--------|
| `analysis.tex` | Full LaTeX source | Teammates can read/review |
| `analysis.pdf` | Compiled PDF (10 pages) | Detailed theoretical derivations |
| `preamble.sty` | LaTeX styling | Technical reference |
| `README.md` | Build instructions | How to compile LaTeX |
| `THEORETICAL_SUMMARY.md` | This file | Quick reference (you are here) |

---

## Action Items by Milestone 2 Deadline

- [x] **Lamaan (Theoretical Analysis) - COMPLETE**
  - [x] Derive PRAM complexity bounds
  - [x] Calculate Amdahl's Law predictions for 4-node system
  - [x] Identify bottlenecks and optimization strategies
  - [x] Document all findings in LaTeX

- [ ] **Muhammad (IPC Implementation) - IN PROGRESS**
  - [ ] Implement shared memory IPC version
  - [ ] Benchmark against network version
  - [ ] Aim for S(4) > 3.0×

- [ ] **Talha (Benchmarking) - IN PROGRESS**
  - [ ] Run all three configurations
  - [ ] Collect per-phase timing data
  - [ ] Prepare results for Sharjeel

- [ ] **Sharjeel (Validation) - BLOCKED** (waiting for Talha's data)
  - [ ] Compare empirical vs. theoretical speedup
  - [ ] Analyze deviations
  - [ ] Validate or refine model

---

## Questions to Investigate

1. **Communication Overhead:** Is 20:1 communication-to-computation ratio realistic? Can non-blocking MPI reduce it?
2. **Scaling Beyond 4 Nodes:** What happens with 6 nodes? Does efficiency drop as predicted?
3. **Network Technology:** How much faster is Bramble (PoE+ single switch) vs. Ethernet? vs. WiFi?
4. **Image Size Scaling:** How do results change for 256×256, 512×512, vs. 1024×1024?
5. **Pure Python vs. NumPy:** Should we profile computation vs. network separately?

---

**Prepared by:** Lamaan Ali Bukhsh  
**For:** Muhammad Fahad, Talha Mudassar, Sharjeel Chandna  
**Date:** April 20, 2026  
**Status:** Ready for Implementation & Benchmarking
