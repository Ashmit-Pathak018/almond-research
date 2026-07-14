# Almond Directory Migration Script
# Run from: C:\Users\ASMIT\Almond\
# Usage: .\migrate.ps1
# Safe: creates backup first, never deletes without confirmation

$Root = "C:\Users\ASMIT\Almond"
$Backup = "$Root\_migration_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  ALMOND DIRECTORY MIGRATION" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Root:   $Root"
Write-Host "Backup: $Backup"
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Back up key files before touching anything"
Write-Host "  2. Create new folder structure"
Write-Host "  3. Move files to their new locations"
Write-Host "  4. Archive historical results"
Write-Host "  5. Ask before deleting stale DB files"
Write-Host ""
$confirm = Read-Host "Proceed? (yes/no)"
if ($confirm -ne "yes") { Write-Host "Aborted." -ForegroundColor Red; exit }

Set-Location $Root

# ── STEP 0: Backup ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[0/6] Creating backup..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $Backup -Force | Out-Null

# Back up the root-level core files (the ones we'll be moving)
$backupFiles = @(
    "almond.py", "memory_block.py", "memory_controller_v2.py",
    "memory_store.py", "judge_v2.py", "eval_unified.py", "server.py"
)
foreach ($f in $backupFiles) {
    if (Test-Path "$Root\$f") {
        Copy-Item "$Root\$f" "$Backup\$f" -Force
    }
}
Write-Host "  Backup saved to: $Backup" -ForegroundColor Green

# ── STEP 1: Create new folders ────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/6] Creating folder structure..." -ForegroundColor Cyan

$folders = @(
    "core",
    "core\memory_pipeline_v2",
    "eval",
    "eval\research",
    "archive",
    "archive\longmem_eval_results",
    "archive\results",
    "archive\Results_folder",
    "archive\xray_runs",
    "apps",
    "desktop",
    "runs"
)
foreach ($f in $folders) {
    New-Item -ItemType Directory -Path "$Root\$f" -Force | Out-Null
    Write-Host "  Created: $f" -ForegroundColor Gray
}
Write-Host "  Done." -ForegroundColor Green

# ── STEP 2: Move core pipeline files into core/ ───────────────────────────────
Write-Host ""
Write-Host "[2/6] Moving core pipeline files to core/..." -ForegroundColor Cyan

$coreFiles = @(
    "almond.py",
    "memory_block.py",
    "memory_controller_v2.py",
    "memory_store.py",
    "judge_v2.py",
    "server.py"
)
foreach ($f in $coreFiles) {
    if (Test-Path "$Root\$f") {
        Move-Item "$Root\$f" "$Root\core\$f" -Force
        Write-Host "  Moved: $f -> core\$f" -ForegroundColor Gray
    } else {
        Write-Host "  Skip (not found): $f" -ForegroundColor DarkGray
    }
}

# Move memory_pipeline_v2 contents into core\memory_pipeline_v2\
Write-Host ""
Write-Host "  Moving memory_pipeline_v2\ files into core\memory_pipeline_v2\..." -ForegroundColor Gray
$pipelineFiles = @(
    "comparison_retriever.py", "entity_extractor.py", "fact_extractor.py",
    "memory_classifier.py", "memory_consolidator.py", "memory_hygiene.py",
    "query_analyzer.py", "ranking_engine.py", "temporal_retriever.py",
    "timeline_index.py"
)
foreach ($f in $pipelineFiles) {
    $src = "$Root\memory_pipeline_v2\$f"
    if (Test-Path $src) {
        Move-Item $src "$Root\core\memory_pipeline_v2\$f" -Force
        Write-Host "  Moved: memory_pipeline_v2\$f -> core\memory_pipeline_v2\$f" -ForegroundColor Gray
    }
}

# Remove now-empty memory_pipeline_v2 dir if empty
if ((Get-ChildItem "$Root\memory_pipeline_v2" -ErrorAction SilentlyContinue).Count -eq 0) {
    Remove-Item "$Root\memory_pipeline_v2" -Force
    Write-Host "  Removed empty: memory_pipeline_v2\" -ForegroundColor Gray
}

Write-Host "  Done." -ForegroundColor Green

# ── STEP 3: Move eval files into eval/ ────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Moving eval files to eval/..." -ForegroundColor Cyan

if (Test-Path "$Root\eval_unified.py") {
    Move-Item "$Root\eval_unified.py" "$Root\eval\eval_unified.py" -Force
    Write-Host "  Moved: eval_unified.py -> eval\eval_unified.py" -ForegroundColor Gray
}

# Copy judge_v2 into eval too (it's already in core/, eval needs it too)
if (Test-Path "$Root\core\judge_v2.py") {
    Copy-Item "$Root\core\judge_v2.py" "$Root\eval\judge_v2.py" -Force
    Write-Host "  Copied: core\judge_v2.py -> eval\judge_v2.py" -ForegroundColor Gray
}

# Move research/ into eval/research/ if it exists
if (Test-Path "$Root\research") {
    Move-Item "$Root\research" "$Root\eval\research" -Force
    Write-Host "  Moved: research\ -> eval\research\" -ForegroundColor Gray
}

Write-Host "  Done." -ForegroundColor Green

# ── STEP 4: Archive historical results ────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Archiving historical results..." -ForegroundColor Cyan

$archiveMoves = @(
    @{ Src="longmem_eval_results"; Dst="archive\longmem_eval_results" },
    @{ Src="results";              Dst="archive\results" },
    @{ Src="Results_folder";       Dst="archive\Results_folder" },
    @{ Src="xray_runs";            Dst="archive\xray_runs" }
)
foreach ($m in $archiveMoves) {
    if (Test-Path "$Root\$($m.Src)") {
        # Move contents rather than the folder itself to avoid nesting issues
        Get-ChildItem "$Root\$($m.Src)" | Move-Item -Destination "$Root\$($m.Dst)" -Force
        Remove-Item "$Root\$($m.Src)" -Force -Recurse
        Write-Host "  Archived: $($m.Src)\ -> $($m.Dst)\" -ForegroundColor Gray
    }
}

# Archive dead eval files
$deadFiles = @("eval_longmem.py", "eval_runner.py", "eval_unified_backup.py")
foreach ($f in $deadFiles) {
    if (Test-Path "$Root\$f") {
        Move-Item "$Root\$f" "$Root\archive\$f" -Force
        Write-Host "  Archived: $f -> archive\$f" -ForegroundColor Gray
    }
}

# Archive misc
foreach ($f in @("xray.py", ".almond_run_history.json")) {
    if (Test-Path "$Root\$f") {
        Move-Item "$Root\$f" "$Root\archive\$f" -Force
        Write-Host "  Archived: $f" -ForegroundColor Gray
    }
}

Write-Host "  Done." -ForegroundColor Green

# ── STEP 5: Move apps and desktop db ─────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Moving apps and desktop DB..." -ForegroundColor Cyan

if (Test-Path "$Root\almond_lab") {
    Move-Item "$Root\almond_lab" "$Root\apps\almond_lab" -Force
    Write-Host "  Moved: almond_lab\ -> apps\almond_lab\" -ForegroundColor Gray
}
if (Test-Path "$Root\almond_desktop.db") {
    Move-Item "$Root\almond_desktop.db" "$Root\desktop\almond_desktop.db" -Force
    Write-Host "  Moved: almond_desktop.db -> desktop\almond_desktop.db" -ForegroundColor Gray
}
if (Test-Path "$Root\stress_test.py") {
    Move-Item "$Root\stress_test.py" "$Root\archive\stress_test.py" -Force
    Write-Host "  Archived: stress_test.py" -ForegroundColor Gray
}
if (Test-Path "$Root\retrieval_pipeline_v2.py") {
    Move-Item "$Root\retrieval_pipeline_v2.py" "$Root\archive\retrieval_pipeline_v2.py" -Force
    Write-Host "  Archived: retrieval_pipeline_v2.py (old, superseded)" -ForegroundColor Gray
}
if (Test-Path "$Root\judge_diagnostic.py") {
    Move-Item "$Root\judge_diagnostic.py" "$Root\archive\judge_diagnostic.py" -Force
    Write-Host "  Archived: judge_diagnostic.py" -ForegroundColor Gray
}

Write-Host "  Done." -ForegroundColor Green

# ── STEP 6: Delete stale DB files (with confirmation) ────────────────────────
Write-Host ""
Write-Host "[6/6] Stale DB files to delete..." -ForegroundColor Cyan
Write-Host "  These are leftover from the last eval run and are auto-recreated:" -ForegroundColor Yellow

$staleDBs = @(
    "almond.db",            # old June 6 db
    "almond_audit.db",      # leftover
    "almond_timeline.db",   # leftover
    "longmem_almond.db",    # leftover from last eval
    "test.py"               # 71 bytes, stub
)

$toDelete = @()
foreach ($f in $staleDBs) {
    if (Test-Path "$Root\$f") {
        $size = (Get-Item "$Root\$f").Length
        Write-Host "  $f ($([math]::Round($size/1024))KB)" -ForegroundColor Gray
        $toDelete += "$Root\$f"
    }
}

if ($toDelete.Count -gt 0) {
    $del = Read-Host "  Delete these $($toDelete.Count) stale files? (yes/no)"
    if ($del -eq "yes") {
        foreach ($f in $toDelete) {
            Remove-Item $f -Force
            Write-Host "  Deleted: $(Split-Path $f -Leaf)" -ForegroundColor DarkGray
        }
        Write-Host "  Done." -ForegroundColor Green
    } else {
        Write-Host "  Skipped deletion." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Nothing to delete." -ForegroundColor Green
}

# ── Final summary ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  MIGRATION COMPLETE" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "New structure:" -ForegroundColor White
Write-Host "  core\                 — pipeline source files"
Write-Host "  core\memory_pipeline_v2\ — retrievers, classifiers, etc"
Write-Host "  eval\                 — eval_unified.py + research runner"
Write-Host "  data\                 — datasets (unchanged)"
Write-Host "  .eval_cache\          — ingestion cache (unchanged)"
Write-Host "  runs\                 — future experiment outputs"
Write-Host "  archive\              — historical results, old files"
Write-Host "  apps\almond_lab\      — Streamlit app"
Write-Host "  desktop\              — desktop db"
Write-Host "  retired\              — unchanged"
Write-Host ""
Write-Host "Backup at: $Backup" -ForegroundColor Yellow
Write-Host ""
Write-Host "NEXT STEP: Update sys.path in eval_unified.py and runner.py" -ForegroundColor Yellow
Write-Host "  Change:  sys.path.insert(0, '.')" -ForegroundColor Gray
Write-Host "  To:      sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))" -ForegroundColor Gray
Write-Host ""
