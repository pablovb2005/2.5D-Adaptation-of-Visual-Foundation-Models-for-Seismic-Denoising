param(
    [string]$ProjectRoot,
    [string]$PanelsDir,
    [string]$OutBaseDir,
    [ValidateSet("ms_ssim_2d", "input_psnr", "input_ms_ssim", "input_mse")]
    [string]$StratifyMetric = "input_ms_ssim",
    [int]$TopExamples = 20,
    [switch]$PooledOnly
)

# Runs analyze_mechanism.py on each per-split CSV (ds101, ds202, ds303) and the
# pooled CSV. Per-split runs give the replication spread; the pooled run gives
# the headline numbers with ~3765 samples. By default, bins are defined from the
# noisy center slice's MS-SSIM against the clean target, not from 2D model output.
# Pass -StratifyMetric ms_ssim_2d only when reproducing the old 2D-baseline split.
#
# Prerequisites: generate_comparison_panels_pooled.ps1 must have been run first.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $PanelsDir) {
    $PanelsDir = Join-Path $ProjectRoot "experiments\summaries\comparison_panels"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$Script = Join-Path $DinoRoot "src\evaluation\analyze_mechanism.py"

$StratFolder = if ($StratifyMetric -eq "ms_ssim_2d") { "by_2d_ms_ssim" } else { "by_$StratifyMetric" }
if (-not $OutBaseDir) {
    $OutBaseDir = Join-Path $ProjectRoot "experiments\summaries\mechanism_analysis\stratification_multidata"
}
$BaseOut = $OutBaseDir

function Run-Mechanism {
    param([string]$Label, [string]$MetadataPath, [string]$OutPath)
    if (-not (Test-Path $MetadataPath)) {
        Write-Host "[$Label] Metadata not found: $MetadataPath -- skipping"
        return
    }
    Write-Host ""
    Write-Host "=== $Label ==="
    Write-Host "  Metadata: $MetadataPath"
    Write-Host "  Output:   $OutPath"
    & $Python $Script `
        "--project-root", $ProjectRoot `
        "--metadata",     $MetadataPath `
        "--out-dir",      $OutPath `
        "--stratify-metric", $StratifyMetric `
        "--top-examples", $TopExamples
}

if (-not $PooledOnly) {
    foreach ($ds in @(101, 202, 303)) {
        Run-Mechanism `
            -Label    "data_seed=$ds" `
            -MetadataPath (Join-Path $PanelsDir "ds$ds\comparison_metadata.csv") `
            -OutPath  (Join-Path $BaseOut "ds$ds\$StratFolder")
    }
}

Run-Mechanism `
    -Label    "pooled (all splits)" `
    -MetadataPath (Join-Path $PanelsDir "pooled\comparison_metadata.csv") `
    -OutPath  (Join-Path $BaseOut "pooled\$StratFolder")

Write-Host ""
Write-Host "Done. Per-split and pooled stratification outputs written to:"
Write-Host "  $BaseOut"
