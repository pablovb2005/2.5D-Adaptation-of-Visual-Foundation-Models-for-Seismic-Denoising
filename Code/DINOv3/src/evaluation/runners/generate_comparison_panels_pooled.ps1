param(
    [string]$ProjectRoot,
    [string]$OutDir,
    [int[]]$TrainingSeeds = @(42, 43, 44),
    [int]$Stride = 5,
    [switch]$NoMidSliceFilter
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "experiments\summaries\comparison_panels"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$Script = Join-Path $DinoRoot "src\evaluation\generate_comparison_panels.py"
$ScriptArgs = @(
    $Script,
    "--project-root", $ProjectRoot,
    "--out-dir", $OutDir,
    "--training-seeds"
) + ($TrainingSeeds | ForEach-Object { "$_" }) + @(
    "--pool-data-seeds",
    "--stride", $Stride
)
if ($NoMidSliceFilter) {
    $ScriptArgs += "--no-mid-slice-filter"
}

Write-Host "Using Python:     $Python"
Write-Host "Project root:     $ProjectRoot"
Write-Host "Output dir:       $OutDir"
Write-Host "Training seeds:   $($TrainingSeeds -join ', ')"
Write-Host "Stride:           $Stride"
Write-Host ""
Write-Host "Outputs written to:"
Write-Host "  $OutDir\ds101\comparison_metadata.csv"
Write-Host "  $OutDir\ds202\comparison_metadata.csv"
Write-Host "  $OutDir\ds303\comparison_metadata.csv"
Write-Host "  $OutDir\pooled\comparison_metadata.csv"

& $Python @ScriptArgs
