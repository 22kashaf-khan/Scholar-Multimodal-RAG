param(
    [ValidateSet("dev","full","down","logs","start")]
    [string]$Mode = "dev"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$ROOT = $PSScriptRoot

# Check Docker is running
docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [!] Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    Write-Host ""
    exit 1
}

# Ensure .env exists
$envFile    = Join-Path $ROOT ".env"
$envExample = Join-Path $ROOT ".env.example"

if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host ""
        Write-Host "  [!] .env was missing - created from .env.example." -ForegroundColor Yellow
        Write-Host "      Set GOOGLE_API_KEY in .env before continuing." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    } else {
        Write-Error ".env not found. Create it from .env.example and set your API keys."
        exit 1
    }
}

# Check at least one LLM API key is set
$groqKey   = Get-Content $envFile | Where-Object { $_ -match "^GROQ_API_KEY=.+" }
$googleKey = Get-Content $envFile | Where-Object { $_ -match "^GOOGLE_API_KEY=.+" }
$openaiKey = Get-Content $envFile | Where-Object { $_ -match "^OPENAI_API_KEY=.+" }
if (-not ($groqKey -or $googleKey -or $openaiKey)) {
    Write-Host ""
    Write-Host "  [!] No LLM API key found in .env" -ForegroundColor Red
    Write-Host "      Set one of: GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Set-Location $ROOT

switch ($Mode) {
    "down" {
        Write-Host "Stopping all containers..." -ForegroundColor Cyan
        docker compose --profile dev --profile full down
        Write-Host "Done." -ForegroundColor Green
    }
    "logs" {
        docker compose --profile dev logs --follow --tail=50
    }
    "start" {
        Write-Host ""
        Write-Host "  Starting existing containers (no rebuild)..." -ForegroundColor Cyan
        Write-Host "  Chat UI  ->  http://localhost:8501"           -ForegroundColor Green
        Write-Host "  API docs ->  http://localhost:8000/docs"      -ForegroundColor Green
        Write-Host ""
        docker compose --profile dev up --remove-orphans
    }
    default {
        Write-Host ""
        Write-Host "  ==========================================" -ForegroundColor Cyan
        Write-Host "   Production RAG Stack  --  $Mode mode"     -ForegroundColor Cyan
        Write-Host "  ==========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Building and starting services..." -ForegroundColor Gray
        Write-Host "  (First run ~2 min while pip deps install)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Once healthy, open:" -ForegroundColor Green
        Write-Host "    Chat UI  ->  http://localhost:8501"      -ForegroundColor Green
        Write-Host "    API docs ->  http://localhost:8000/docs" -ForegroundColor Green
        if ($Mode -eq "full") {
            Write-Host "    Grafana  ->  http://localhost:3000  (admin / admin)" -ForegroundColor Green
            Write-Host "    Prometheus-> http://localhost:9090"  -ForegroundColor Green
        }
        Write-Host ""
        Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
        Write-Host ""

        docker compose --profile $Mode up --build --remove-orphans
    }
}
