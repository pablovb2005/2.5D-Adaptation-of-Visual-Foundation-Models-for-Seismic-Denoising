param(
    [string]$ProjectRoot,
    [string]$OutDir,
    [int]$DataSeed = 101,
    [int]$TrainingSeed = 42,
    [int]$NPanels = 4,
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
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Script = Join-Path $DinoRoot "src\evaluation\generate_comparison_panels.py"
$ScriptArgs = @(
    $Script,
    "--project-root", $ProjectRoot,
    "--out-dir", $OutDir,
    "--data-seed", $DataSeed,
    "--training-seed", $TrainingSeed,
    "--n-panels", $NPanels,
    "--stride", $Stride
)
if ($NoMidSliceFilter) {
    $ScriptArgs += "--no-mid-slice-filter"
}

Write-Host "Using Python:    $Python"
Write-Host "Project root:    $ProjectRoot"
Write-Host "Output dir:      $OutDir"
Write-Host "Data seed:       $DataSeed"
Write-Host "Training seed:   $TrainingSeed"
Write-Host "Panels:          $NPanels"

& $Python @ScriptArgs
