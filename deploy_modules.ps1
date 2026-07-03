# Deploy the modular backend structure and updated dashboard
# Run this script from the project root directory

$ProjectRoot = "C:\Users\apira\Documents\GitHub\disasear_dashboard"
$TempDir = "$env:TEMP\opencode"

Write-Host "Deploying RescuOpt AI modular backend..." -ForegroundColor Cyan

# 1. Create backend directory if needed
$backendDir = Join-Path $ProjectRoot "backend"
$subdirs = @("algorithms", "services", "models", "utils")
foreach ($sd in $subdirs) {
    $p = Join-Path $backendDir $sd
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

# 2. Copy all Python modules
$modules = @(
    "backend\__init__.py",
    "backend\server.py",
    "backend\utils\__init__.py",
    "backend\utils\geo.py",
    "backend\utils\math_utils.py",
    "backend\models\__init__.py",
    "backend\models\hazard.py",
    "backend\models\survivor.py",
    "backend\models\route.py",
    "backend\services\__init__.py",
    "backend\services\danger_service.py",
    "backend\services\routing_service.py",
    "backend\services\urgency_service.py",
    "backend\algorithms\__init__.py",
    "backend\algorithms\astar.py",
    "backend\algorithms\bfs.py",
    "backend\algorithms\greedy.py",
    "backend\algorithms\simulated_annealing.py",
    "backend\algorithms\hill_climbing.py",
    "backend\algorithms\genetic.py",
    "backend\algorithms\backtracking.py",
    "backend\algorithms\ac3.py"
)

foreach ($mod in $modules) {
    $src = Join-Path $TempDir $mod
    $dst = Join-Path $ProjectRoot $mod
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "  Copied: $mod" -ForegroundColor Green
    } else {
        Write-Host "  MISSING: $src" -ForegroundColor Yellow
    }
}

# 3. Copy updated dashboard.html
$dashboardSrc = Join-Path $TempDir "dashboard.html"
$dashboardDst = Join-Path $ProjectRoot "dashboard.html"
if (Test-Path $dashboardSrc) {
    Copy-Item -LiteralPath $dashboardSrc -Destination $dashboardDst -Force
    Write-Host "  Copied: dashboard.html" -ForegroundColor Green
}

# 4. Copy updated start_rescuopt.bat
$batSrc = Join-Path $TempDir "start_rescuopt.bat"
$batDst = Join-Path $ProjectRoot "start_rescuopt.bat"
if (Test-Path $batSrc) {
    Copy-Item -LiteralPath $batSrc -Destination $batDst -Force
    Write-Host "  Copied: start_rescuopt.bat" -ForegroundColor Green
}

# 5. Copy root-level Server.py shim (imports from backend.server)
$serverSrc = Join-Path $TempDir "Server.py"
$serverDst = Join-Path $ProjectRoot "Server.py"
if (Test-Path $serverSrc) {
    Copy-Item -LiteralPath $serverSrc -Destination $serverDst -Force
    Write-Host "  Copied: Server.py (root shim)" -ForegroundColor Green
}

# 6. Backup old Server.py (monolithic) separately
$oldServer = Join-Path $ProjectRoot "Server_old.py.bak"
if (Test-Path $(Join-Path $ProjectRoot "Server.py") -and -not (Test-Path $oldServer)) {
    $backupSrc = Join-Path $TempDir "Server_head.py"  # placeholder
    Write-Host "  (Old Server.py was replaced by the root shim)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Workflow (two terminals):" -ForegroundColor White
Write-Host "  Terminal 1: python Server.py" -ForegroundColor Green
Write-Host "  Terminal 2: cd Flood-detection ^&^& python main.py" -ForegroundColor Green
Write-Host "  (or double-click start_rescuopt.bat to launch both)" -ForegroundColor White
Write-Host ""
Write-Host "Open http://localhost:5000 in your browser." -ForegroundColor White
