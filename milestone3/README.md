# Milestone 3 - Distributed MPI Pipeline (Docker, 4 nodes)

Roadmap reference: **M3 - Physical Build & Distributed MPI** (`PDCRoadMap.pdf`).
This milestone is delivered on the existing 4-container Docker cluster
(`n1` master + `n2`, `n3`, `n4` workers) instead of physical Raspberry Pis.
Each container is pinned to **1 CPU / 2 GB RAM**, so each rank still maps
1:1 with a Pi-class compute node.

## Roadmap Tasks → Files

| # | Roadmap task | Where it lives |
|---|---|---|
| 01 | Flash OS / SD cards | **N/A in Docker** (replaced by `docker/Dockerfile`) |
| 02 | Network config / static IPs | `compose/docker-compose.yml` (`LABnet` bridge, fixed hostnames) |
| 03 | SSH passwordless auth | Already shipped in M0 (`docker/Dockerfile`, `docker/entrypoint.sh`) |
| 04 | `setup.sh` uniform environment | Replaced by `docker/Dockerfile` (pinned `mpi4py==4.0.1`, `numpy==1.26.4`, `pillow==10.4.0`) |
| 05 | Sync code + Hello World MPI | Already shipped in M0 (`mpi/hello_mpi.py`, `scripts/m0-run-tests.ps1`) |
| 06 | **Distributed Histogram Equalization** (Scatter/Gather) | `milestone3/app/histogram_eq_mpi.py` |
| 07 | **Non-divisible image sizes via `Scatterv`** | `milestone3/app/chunking.py`, same `histogram_eq_mpi.py` |
| 08 | **Re-run benchmarks at P = 1, 2, 3, 4** | `milestone3/app/benchmark_mpi.py`, `scripts/m3-bench.ps1` |

## Layout

```
milestone3/
├── README.md                         # this file
├── MILESTONE3_REPORT.md              # exit-criteria status + verification of M0-M2
├── app/
│   ├── chunking.py                   # compute_chunks(M, P) + Scatterv helpers
│   ├── histogram_eq_serial.py        # ground-truth single-process reference
│   ├── histogram_eq_mpi.py           # distributed pipeline (Scatterv/Allreduce/Gatherv)
│   ├── bench_mpi_run.py              # one mpirun-driven worker invocation
│   ├── benchmark_mpi.py              # spawns mpirun for P=1..4, writes CSV
│   └── generate_test_images.py       # deterministic test images (incl. 1025x1025)
├── tests/
│   ├── test_chunking.py              # unit tests, no MPI required
│   ├── test_serial_eq.py             # unit tests for the reference equaliser
│   └── test_distributed_eq.py        # MPI sweep (run via mpirun -np 4)
├── data/                             # generated PNG inputs (built on demand)
└── output/                           # written by --output flags / CLI runs
```

## How to run (PowerShell, from repo root)

```powershell
# 1. Bring the cluster up (idempotent).
powershell -ExecutionPolicy Bypass -File .\scripts\m3-up.ps1

# 2. Generate the deterministic test images inside the cluster.
powershell -ExecutionPolicy Bypass -File .\scripts\m3-prepare-data.ps1

# 3. Run the unit + integration tests (PASS expected on every shape).
powershell -ExecutionPolicy Bypass -File .\scripts\m3-run-tests.ps1

# 4. Speedup sweep at P = 1, 2, 3, 4 -> CSV under results/.
powershell -ExecutionPolicy Bypass -File .\scripts\m3-bench.ps1

# 5. Optional: benchmark a non-divisible image to exercise the Scatterv path.
powershell -ExecutionPolicy Bypass -File .\scripts\m3-bench.ps1 `
    -Image '/workspace/milestone3/data/test_1025x1025_nondiv4.png'
```

## Algorithm summary (distributed histogram equalisation)

1. **Geometry broadcast** - rank 0 broadcasts `(height, width)` so each
   rank can size its receive buffer.
2. **Scatterv** - row blocks computed by `compute_chunks(M, P)` are
   distributed; remainder rows go to the first `M % P` ranks. Counts and
   displacements are computed once, in element units, by
   `chunks_to_scatterv`.
3. **Local histogram** - `np.histogram(local, bins=256, range=(0,256))`
   on the assigned slice only.
4. **Allreduce** - `MPI.SUM` of local histograms gives every rank the
   same global histogram (avoids a separate `Bcast` of the mapping).
5. **Mapping** - identical 256-entry uint8 lookup table built on every
   rank from the global histogram (CDF + linear remap).
6. **Apply + Gatherv** - each rank applies `mapping[local]`; rank 0
   reassembles the full image with `Gatherv` using the same counts.

The pipeline validates byte-for-byte against the serial NumPy reference
in `histogram_eq_serial.py` for every shape in `test_distributed_eq.py`.

## Why neutral hostnames (`n1..n4`) instead of `master/worker1..3`

This was already documented in `README_M0.md` and is preserved here:
keeping leadership *logical* (rank 0 = current master) makes future M4
work on leader election (the bully algorithm) easier - we never need to
rename containers when the master changes.

## What is **not** covered by this milestone

- M3 task 08 mentions "Re-run M1 multiprocessing benchmarks on a single
  Pi". Because each Docker container is pinned to 1 CPU, in-container
  multiprocessing speedups are bounded by 1× and not informative. The
  equivalent comparison here is the **MPI** sweep at P = 1..4 produced
  by `benchmark_mpi.py` - it is the right hardware-parity number.
- Hardware-only failure modes (SD-card corruption, IP changes after
  reboot, etc.) listed in the roadmap are not applicable to Docker.

## Carry-over items from M1 / M2 noticed during verification

These are **not blockers** for M3, but should be addressed before the M5
final report (see `MILESTONE3_REPORT.md` for full status):

- M1 task 05: a CIFAR-10 loader is referenced in the roadmap but not
  present in `milestone1/`. Test images are currently synthetic.
- M1 task 06: there is a markdown table for threading speedup but no
  `matplotlib` chart artifact under `milestone1/`.
- M2 task 03: empirical α and β network measurements (ping / iperf3
  inside Docker) are not captured anywhere in `milestone2/` - the
  THEORETICAL_SUMMARY uses estimated values only.
- M2 task 05: the three-way comparison chart (threading vs.
  multiprocessing vs. shared memory) is referenced in the report draft
  but not committed as a PNG / CSV.
