# Bring up the 4-node Docker cluster used for Milestone 3 work.
# Reuses the same compose file as M0; safe to run repeatedly.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\m3-up.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

docker compose -f compose/docker-compose.yml up -d --build --remove-orphans
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed.' }

docker compose -f compose/docker-compose.yml ps
$running = docker compose -f compose/docker-compose.yml ps --status running --services |
    Measure-Object | Select-Object -ExpandProperty Count
if ($running -lt 4) { throw "Expected 4 running services, found $running." }

Write-Host "M3 cluster is up with 4 running nodes."
