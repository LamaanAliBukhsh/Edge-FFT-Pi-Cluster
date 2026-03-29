# Starts and builds the 4-node virtual cluster for Milestone 0.
# If script policy blocks execution, run:
# powershell -ExecutionPolicy Bypass -File .\scripts\m0-up.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

docker compose -f compose/docker-compose.yml up -d --build --remove-orphans
if ($LASTEXITCODE -ne 0) {
	throw 'docker compose up failed.'
}

docker compose -f compose/docker-compose.yml ps
$runningCount = docker compose -f compose/docker-compose.yml ps --status running --services | Measure-Object | Select-Object -ExpandProperty Count
if ($runningCount -lt 4) {
	throw "Expected 4 running services, found $runningCount."
}

Write-Host "M0 cluster is up with 4 running nodes."
