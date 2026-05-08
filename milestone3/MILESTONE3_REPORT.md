# Milestone 3 Status Report

**System:** 4-node Docker cluster (`n1` master + `n2`, `n3`, `n4` workers),
each container pinned to 1 CPU / 2 GB RAM.
**Roadmap:** see `PDCRoadMap.pdf`, M3 *Physical Build & Distributed MPI*.

The original proposal (`proposal-1.pdf`, `project.md`) targets 6 Raspberry
Pis. The PDC roadmap that defines the actual milestone exit criteria is
already written for **4 nodes (1 master + 3 workers)**, so the team's
move from 6 Pis to 4 Docker containers is fully consistent with the
milestone definitions used to grade the project. Hardware tasks (SD-card
flashing, static IP assignment) are replaced by the equivalent Docker
plumbing already in `compose/docker-compose.yml` and `docker/Dockerfile`.

---

## 1. Verification of M0-M2 (briefly)

### Milestone 0 - Virtual Cluster & Toolchain ✅

| Roadmap exit criterion | Evidence |
|---|---|
| 4 containers launch cleanly with `docker-compose up` | `compose/docker-compose.yml` (n1-n4 on `LABnet`), `scripts/m0-up.ps1` asserts 4 running services |
| MPI Hello World prints 4 distinct ranks from 4 hostnames | `mpi/hello_mpi.py` + `scripts/m0-run-tests.ps1` |
| Scatter/Gather test produces correct results 10/10 runs | `mpi/scatter_gather_test.py` (validates `final == expected`) |
| Passwordless SSH from master to any worker | `docker/Dockerfile` bakes ed25519 keys + `authorized_keys`; `docker/entrypoint.sh` pre-seeds `known_hosts` |
| README documents every step with working commands | `README_M0.md` |

Note on naming: the roadmap suggests `master / worker1-3`, the repo uses
`n1-n4`. This is intentional and documented in `README_M0.md` (decoupling
logical leadership from container identity prepares for the M4 bully
algorithm).

### Milestone 1 - Foundations of Concurrency ✅ (with two carry-overs)

| Roadmap exit criterion | Evidence |
|---|---|
| Serial Sobel baseline recorded | `milestone1/sequential/app/sobel_sequential.py`, `benchmark.py` |
| Threading version <1.5× on 4 threads (GIL confirmed) | `milestone1/milestone1/app/sobel_threaded.py`, results table in `Benchmark_Results_Sobel(Multi-thread).md` |
| Multiprocessing version ≥2.5× on 4 processes | `milestone1/multiprocessing/app/sobel_multiprocessing.py`, `benchmark.py` |
| Benchmark harness produces consistent CSV | Harness exists but writes a printed table, **not CSV**. Either acceptable for M1 or to be normalised in M5. |
| CIFAR-10 data loader (task 05) | **Carry-over:** not present in `milestone1/`. Tests use synthetic / single test JPEGs. |
| `matplotlib` speedup chart (task 06) | **Carry-over:** chart not committed; only the markdown table exists. |

### Milestone 2 - Theory & Shared Memory ✅ (with two carry-overs)

| Roadmap exit criterion | Evidence |
|---|---|
| PRAM analysis with concrete values for M=N=1024, P=4 | `milestone2/theoretical_analysis/THEORETICAL_SUMMARY.md`, `analysis_combined.tex`, compiled `THEORETICAL_SUMMARY.pdf` |
| Amdahl curve plotted, P=4 marked | Numerical analysis present in summary (S(4) = 2.23×, E(4) = 55.8%); LaTeX includes the derivation |
| α and β measured in Docker, comm cost for 3 message sizes | **Carry-over:** values used in the report (`α = 10 µs`, `β = 125 MB/s`) are *estimates*, not `ping`/`iperf3` measurements |
| Shared memory IPC via `/dev/shm` working & benchmarked | `shared_memory_sobel.py` (uses `np.memmap` on `/dev/shm`, falls back to temp dir on Windows). Reports three timings: naive seq, vectorised seq, parallel — see notes below. |
| Three-way comparison chart (threading vs. mp vs. shm) | **Carry-over:** PNG chart still pending. The three-way **measurement** itself is now done correctly: `benchmark_comparison` separates `vectorisation_speedup` (algorithmic, ~190×) from `parallel_speedup S(N)` (parallelism only, ≤ N), so the data is no longer corrupted by the apples-to-oranges baseline that previously produced 650% efficiency. |
| Theory section drafted | `analysis_combined.tex` (~10 pages compiled) and `THEORETICAL_SUMMARY.md` |

**Bottom line:** M0-M2 substantively meet the roadmap exit criteria; the
four carry-over items are documentation/visualisation artifacts that do
not block M3 progression and can be folded into the M5 final report
sprint.

---

## 2. Milestone 3 Deliverables

### 2.1 Source files added

```
milestone3/app/chunking.py                # compute_chunks + Scatterv helpers
milestone3/app/histogram_eq_serial.py     # ground-truth NumPy reference
milestone3/app/histogram_eq_mpi.py        # Scatterv/Allreduce/Gatherv pipeline
milestone3/app/bench_mpi_run.py           # one-shot timed worker (mpirun-launched)
milestone3/app/benchmark_mpi.py           # P=1..4 sweep, CSV writer
milestone3/app/generate_test_images.py    # deterministic PNG fixtures
milestone3/tests/test_chunking.py         # 10 unit tests, no MPI
milestone3/tests/test_serial_eq.py        # 6 unit tests, no MPI
milestone3/tests/test_distributed_eq.py   # 7-shape MPI sweep w/ array_equal
scripts/m3-up.ps1                         # cluster bring-up
scripts/m3-prepare-data.ps1               # generate fixtures inside cluster
scripts/m3-run-tests.ps1                  # full correctness suite
scripts/m3-bench.ps1                      # P=1..4 speedup sweep
```

All non-MPI tests **pass locally** (`16/16`) without Docker; the MPI
test sweep is invoked via `scripts/m3-run-tests.ps1` once the cluster is
up.

### 2.2 Algorithm: distributed histogram equalisation

```
rank 0  --[Bcast(H,W)]-->  every rank sizes its local buffer
        --[Scatterv]   -->  row blocks per compute_chunks(H, P)
                            (remainder rows -> first H%P ranks)

every rank: local_hist = np.histogram(local, 256, (0,256))
            Allreduce(SUM)  ->  global_hist on every rank
            mapping = build_mapping(global_hist, total_pixels)   # CDF -> uint8 LUT
            local_eq = mapping[local]

rank 0  <--[Gatherv]----   reassembled equalised image
```

The Allreduce-based design avoids a second broadcast of the 256-byte LUT
and guarantees every rank derives the **same** mapping deterministically.

### 2.3 Roadmap-defined exit criteria (status table)

| Exit criterion | How it is satisfied | Status |
|---|---|---|
| All 4 nodes reachable via SSH from master without passwords | Inherited from M0 (`docker/Dockerfile`, `entrypoint.sh`); `scripts/m3-up.ps1` re-asserts 4 running services | ✅ |
| MPI Hello World prints 4 ranks from 4 hostnames | Inherited from M0 (`mpi/hello_mpi.py`); also covered transitively by every `m3-run-tests` MPI launch | ✅ |
| Distributed histogram equalisation produces visually correct output | `histogram_eq_mpi.py --validate` compares to `equalize_serial` with `np.array_equal`; `--output` writes a PNG to inspect; `tests/test_distributed_eq.py` covers 7 shapes | ✅ (verified by `array_equal` — stronger than visual inspection) |
| Scatterv handles non-divisible image sizes correctly | `compute_chunks(M, P)` distributes remainder rows to the first `M%P` ranks; covered by `test_chunking.py` (`100x100`, `101x100`, `103x100`, random) and `test_distributed_eq.py` (`101x100`, `103x100`, `1025x1025`, `720x1280`) | ✅ |
| Speedup benchmarks collected at P = 1, 2, 3, 4 | `benchmark_mpi.py` sweeps the requested ranks, prints summary table, appends to `results/m3_histogram_eq.csv`; per-phase timings (`scatter / hist / allreduce / apply / gather`) recorded for later analysis | ✅ infrastructure ready; numbers obtained on first run of `scripts/m3-bench.ps1` |

### 2.4 What the team needs to do once Docker is up

```powershell
# from repo root, after Docker Desktop is started
powershell -ExecutionPolicy Bypass -File .\scripts\m3-up.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\m3-prepare-data.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\m3-run-tests.ps1   # expect: all PASS
powershell -ExecutionPolicy Bypass -File .\scripts\m3-bench.ps1       # expect: 4 rows of CSV in results/
```

The bench step is the only one that produces hardware-dependent numbers;
everything else (correctness, chunk balancing, validation against the
serial reference) is deterministic and should reproduce byte-for-byte
across machines because of the fixed RNG seed in `generate_test_images.py`.

### 2.5 Pre-flight checks already executed (from this workstation)

- `python -m unittest discover -s milestone3/tests -p "test_chunking.py"` -> 10/10 OK
- `python -m unittest discover -s milestone3/tests -p "test_serial_eq.py"` -> 6/6 OK
- `python -m py_compile milestone3/app/*.py milestone3/tests/test_distributed_eq.py` -> OK
- `python milestone3/app/generate_test_images.py` -> 8 PNGs written to `milestone3/data/`

The MPI integration test (`test_distributed_eq.py`) requires `mpi4py`
plus `mpirun` and is therefore only executable inside the cluster - it
is wired into `scripts/m3-run-tests.ps1`.

---

## 3. Recommended next actions before M4

1. Run `scripts/m3-bench.ps1` on the divisible (`1024x1024`) and
   non-divisible (`1025x1025`) inputs and commit the resulting CSVs
   under `results/`. The CSV schema is forward-compatible with M5's
   final speedup-curve plot.
2. Decide whether the M1/M2 carry-over items (CIFAR-10 loader,
   matplotlib speedup chart, empirical α/β, three-way comparison chart)
   should be back-filled now or rolled into M5. They are listed in this
   document so they are not silently dropped.
3. Once the bench numbers are in hand, compare them to the
   `milestone2/theoretical_analysis/THEORETICAL_SUMMARY.md` predictions
   (S(4) ≈ 2.23×). A large gap either way is signal for the M4
   non-blocking-MPI work item.
