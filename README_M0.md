# Milestone 0 (M0) - Virtual 4-Node Pi Cluster with Docker

This setup gives you a local distributed lab that mirrors your planned 4-node Raspberry Pi cluster.

## Why n1, n2, n3, n4 naming is correct
Using neutral names is a good design choice for a fail-safe system.

- Do not hard-code a permanent "master" identity in naming.
- Keep leadership logical (algorithmic), not structural.
- In MPI, rank 0 often coordinates a task, but that is runtime behavior, not a permanent node role.

## What this M0 setup includes
- 4 containers: n1, n2, n3, n4
- Shared ARM-targeted Docker build (`linux/arm64`)
- OpenMPI + mpi4py + SSH for cross-node process launch
- Pi-like constraints per node: 1 CPU, 2 GB RAM
- M0 tests:
  - `hello_mpi.py`
  - `scatter_gather_test.py`
  - `async_ping_test.py`

## Folder map
- `compose/docker-compose.yml`: cluster topology and limits
- `docker/Dockerfile`: runtime image for all nodes
- `docker/entrypoint.sh`: container bootstrap (SSH + keepalive)
- `mpi/hostfile`: optional node list reference (tests currently use explicit `--host`)
- `mpi/*.py`: MPI validation programs
- `scripts/*.ps1`: run helpers

## Run sequence (PowerShell)
Prerequisites:
- Docker Desktop must be running.
- If script execution is restricted, use `powershell -ExecutionPolicy Bypass -File <script>`.
- Optional: set `PDCM0_PLATFORM=linux/arm64` when ARM emulation is available.

Example for ARM parity run:
- `$env:PDCM0_PLATFORM='linux/arm64'`
- `powershell -ExecutionPolicy Bypass -File .\scripts\m0-up.ps1`

1. Build and start cluster
   - `./scripts/m0-up.ps1`
2. Run all M0 tests
   - `./scripts/m0-run-tests.ps1`
3. Check status
   - `./scripts/m0-status.ps1`
4. Stop cluster
   - `./scripts/m0-down.ps1`

## If Docker ARM emulation is not enabled
If `linux/arm64` fails on your machine:

1. Enable Docker Desktop virtualization and emulation support.
2. Keep this project on `linux/arm64` for compatibility with real Pis.
3. As a temporary fallback for local debugging, set platform to host architecture, then switch back before Pi deployment.

## M0 Definition of Done checklist
- [ ] All 4 containers are healthy and reachable by name.
- [ ] `hello_mpi.py` prints all ranks across nodes.
- [ ] `scatter_gather_test.py` reports validation PASS.
- [ ] `async_ping_test.py` completes on all ranks.
- [ ] Team can reproduce setup from scratch using only scripts.
