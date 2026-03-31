#!/usr/bin/env pwsh
# deploy.ps1 — Re-apply all source code changes to running containers.
# Run this after every: docker compose up / .\start.ps1 start

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$root = $PSScriptRoot
$src  = "$root\src\production_rag"

$files = @(
    "core\types.py",
    "vectorstore\schema.py",
    "vectorstore\weaviate_client.py",
    "generation\synthesizer.py",
    "generation\streaming.py",
    "core\llm_client.py",
    "api\routers\chat.py",
    "adaptive\crag.py",
    "ingestion\chunkers\factory.py",
    "ingestion\worker.py",
    "ingestion\loaders\docling_pdf.py",
    "ingestion\chunkers\docling.py"
)

function Copy-If-Exists($localPath, $container, $remotePath) {
    if (Test-Path $localPath) {
        docker cp $localPath "${container}:${remotePath}" 2>&1 | Out-Null
        Write-Host "  OK  $container <- $($localPath.Replace($src + '\', ''))" -ForegroundColor Green
    } else {
        Write-Host "  --  SKIP (not found): $localPath" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "  Deploying code changes to containers..." -ForegroundColor Cyan
Write-Host ""

foreach ($f in $files) {
    $remotePath = "/home/rag/app/src/production_rag/" + $f.Replace('\', '/')
    Copy-If-Exists "$src\$f" "k-worker-1" $remotePath
    Copy-If-Exists "$src\$f" "k-api-1"    $remotePath
}

# Streamlit app
Copy-If-Exists "$root\ui\streamlit_app.py" "k-streamlit-1" "/home/rag/app/ui/streamlit_app.py"

Write-Host ""
Write-Host "  Restarting containers..." -ForegroundColor Cyan
docker restart k-worker-1 k-api-1 k-streamlit-1 | Out-Null

Write-Host ""
Write-Host "  Connecting worker to docling network..." -ForegroundColor Cyan
docker network connect care-gpt_care-gpt-network k-worker-1 2>&1 | Out-Null
# Ignore "already exists" error — connection persists across restarts

Write-Host ""
Write-Host "  Waiting for health checks..." -ForegroundColor Gray
Start-Sleep -Seconds 12
docker ps --format "  {{.Names}}`t{{.Status}}" | Select-String "^  k-"

Write-Host ""
Write-Host "  Done. Open http://localhost:8501" -ForegroundColor Green
Write-Host ""
