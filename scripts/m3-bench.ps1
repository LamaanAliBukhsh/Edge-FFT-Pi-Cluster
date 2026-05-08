# Run the Milestone 3 speedup benchmark across P = 1,2,3,4 from rank 0.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\m3-bench.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\m3-bench.ps1 -Image '/workspace/milestone3/data/test_1025x1025_nondiv4.png'
param(
    [string]$Image = '/workspace/milestone3/data/test_1024x1024_div4.png',
    [string]$Ranks = '1,2,3,4',
    [int]$Runs = 5,
    [int]$Warmup = 1,
    [string]$Csv = '/workspace/results/m3_histogram_eq.csv'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'compose/docker-compose.yml'

# benchmark_mpi.py spawns mpirun internally; it must run from inside n1.
$cmd = "python /workspace/milestone3/app/benchmark_mpi.py " +
       "--image $Image --ranks $Ranks --runs $Runs --warmup $Warmup --csv $Csv"

Write-Host "Running M3 benchmark on image=$Image ranks=$Ranks runs=$Runs"
docker compose -f $compose exec n1 sh -lc $cmd
if ($LASTEXITCODE -ne 0) { throw 'benchmark_mpi.py failed.' }

Write-Host "M3 benchmark complete. CSV: $Csv"
