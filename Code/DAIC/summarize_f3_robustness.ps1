# Regenerate F3 robustness summaries (CSVs, PNGs, report) from local data.
# Run from anywhere inside the project:
#   .\Code\DAIC\summarize_f3_robustness.ps1
#
# Prerequisites: Python with numpy, matplotlib, pyyaml installed.
# Raw result files (f3_metrics.csv) must already be synced under
#   experiments/runs/robustness/f3_allsections/
# Run daic_mux sync first if they are not there yet.

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$robRoot     = Join-Path $projectRoot "experiments\runs\robustness"
$summaryRoot = Join-Path $projectRoot "experiments\summaries"
$pythonPath  = Join-Path $projectRoot "Code\DINOv3\src"
$summarizer  = Join-Path $pythonPath  "evaluation\summarize_robustness.py"

$env:PYTHONPATH = $pythonPath

function Run-Summarizer($expSets, $outLabel) {
    $outDir = Join-Path $summaryRoot "f3_allsections_$outLabel"
    $resultsDir = Join-Path $robRoot "f3_allsections\$expSets"
    if (-not (Test-Path $resultsDir)) {
        Write-Host "  [SKIP] no results yet: $resultsDir"
        return
    }
    $hasResults = Get-ChildItem -Path $resultsDir -Filter "f3_metrics.csv" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $hasResults) {
        Write-Host "  [SKIP] no f3_metrics.csv files found under: $resultsDir"
        return
    }
    Write-Host ""
    Write-Host "=== $expSets ===" -ForegroundColor Cyan
    python $summarizer `
        --robustness-root $robRoot `
        --result-dataset  f3_allsections `
        --experiment-sets $expSets `
        --output-dir      $outDir
}

Write-Host "=== F3 Robustness Summarizer (local) ===" -ForegroundColor Cyan

Run-Summarizer "main_multidata"    "main_multidata"
Run-Summarizer "full_ft_multidata" "full_ft_multidata"

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Results in: $summaryRoot"
