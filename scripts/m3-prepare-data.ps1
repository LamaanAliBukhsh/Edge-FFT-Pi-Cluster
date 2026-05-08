# Generate Milestone 3 test images inside the cluster (rank 0 host n1).
# This must be re-run any time generate_test_images.py changes so every
# container observes the same /workspace/milestone3/data/*.png inputs.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'compose/docker-compose.yml'
docker compose -f $compose exec n1 sh -lc `
    'python /workspace/milestone3/app/generate_test_images.py'
if ($LASTEXITCODE -ne 0) { throw 'generate_test_images.py failed.' }

Write-Host 'M3 test images generated under milestone3/data/.'
