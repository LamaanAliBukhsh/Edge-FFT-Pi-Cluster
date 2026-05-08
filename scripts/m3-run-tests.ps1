# Run all Milestone 3 correctness checks from inside the cluster.
#   1. unit tests for chunking + serial reference (no MPI needed)
#   2. distributed correctness sweep across divisible / non-divisible sizes
#   3. quick end-to-end CLI smoke run with --validate
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\m3-run-tests.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'compose/docker-compose.yml'
$mpiBase = 'mpirun --allow-run-as-root --host n1,n2,n3,n4 -np 4'

Write-Host '--- Unit tests (no MPI): chunking + serial equaliser ---'
docker compose -f $compose exec n1 sh -lc `
    'python -m unittest discover -s /workspace/milestone3/tests -p "test_chunking.py" -v'
if ($LASTEXITCODE -ne 0) { throw 'chunking unit tests failed.' }

docker compose -f $compose exec n1 sh -lc `
    'python -m unittest discover -s /workspace/milestone3/tests -p "test_serial_eq.py" -v'
if ($LASTEXITCODE -ne 0) { throw 'serial equaliser tests failed.' }

Write-Host '--- Distributed correctness sweep across image sizes ---'
docker compose -f $compose exec n1 sh -lc `
    "$mpiBase python /workspace/milestone3/tests/test_distributed_eq.py"
if ($LASTEXITCODE -ne 0) { throw 'distributed equaliser sweep failed.' }

Write-Host '--- End-to-end CLI smoke (1024x1024, --validate) ---'
$smokeCmd = "$mpiBase python /workspace/milestone3/app/histogram_eq_mpi.py " +
            "--image /workspace/milestone3/data/test_1024x1024_div4.png " +
            "--output /workspace/milestone3/output/test_1024x1024_eq.png " +
            "--validate"
docker compose -f $compose exec n1 sh -lc $smokeCmd
if ($LASTEXITCODE -ne 0) { throw 'CLI smoke run failed validation.' }

Write-Host 'All M3 correctness checks passed.'
