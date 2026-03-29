# Stops the Milestone 0 virtual cluster.
# If script policy blocks execution, run:
# powershell -ExecutionPolicy Bypass -File .\scripts\m0-down.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

docker compose -f compose/docker-compose.yml down --remove-orphans
Write-Host "M0 cluster is down."
