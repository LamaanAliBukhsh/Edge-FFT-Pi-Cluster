# Shows container and network status for quick troubleshooting.
# If script policy blocks execution, run:
# powershell -ExecutionPolicy Bypass -File .\scripts\m0-status.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

docker compose -f compose/docker-compose.yml ps
docker network ls | Select-String 'LABnet'
