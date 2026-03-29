# Runs all Milestone 0 MPI validation tests from node N1.
# If script policy blocks execution, run:
# powershell -ExecutionPolicy Bypass -File .\scripts\m0-run-tests.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'compose/docker-compose.yml'
$mpiBase = 'mpirun --allow-run-as-root --host n1,n2,n3,n4 -np 4'

Write-Host 'Running hello_mpi.py ...'
docker compose -f $compose exec n1 sh -lc "$mpiBase python /workspace/mpi/hello_mpi.py"
if ($LASTEXITCODE -ne 0) { throw 'hello_mpi.py failed.' }

Write-Host 'Running scatter_gather_test.py ...'
docker compose -f $compose exec n1 sh -lc "$mpiBase python /workspace/mpi/scatter_gather_test.py"
if ($LASTEXITCODE -ne 0) { throw 'scatter_gather_test.py failed.' }

Write-Host 'Running async_ping_test.py ...'
docker compose -f $compose exec n1 sh -lc "$mpiBase python /workspace/mpi/async_ping_test.py"
if ($LASTEXITCODE -ne 0) { throw 'async_ping_test.py failed.' }

Write-Host 'All M0 tests finished.'
