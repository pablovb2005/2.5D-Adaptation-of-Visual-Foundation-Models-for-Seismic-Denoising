param(
    [string]$ProjectRoot,
    [string]$OutDir,
    [int[]]$TrainingSeeds = @(42, 43, 44),
    [int]$MaxSamples = 64,
    [int]$Stride = 5,
    [int]$DistantStride = 5,
    [ValidateSet("uniform", "top-gap")]
    [string]$Selection = "uniform",
    [switch]$FullTest
)

# Runs context counterfactuals for all 3 data seeds, averaging over all 3 training
# seeds per split. Writes per-split outputs (ds101_ts42_43_44/, ...) and a pooled
# output (pooled_ts42_43_44/).
#
# Prerequisites: generate_comparison_panels_pooled.ps1 must have been run first so
# that per-split comparison_metadata.csv files exist (needed for top-gap selection).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "experiments\summaries\mechanism_analysis\context_counterfactuals"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$Script = Join-Path $DinoRoot "src\evaluation\analyze_context_counterfactuals.py"

$ScriptArgs = @(
    $Script,
    "--project-root",   $ProjectRoot,
    "--out-dir",        $OutDir,
    "--training-seeds"
) + ($TrainingSeeds | ForEach-Object { "$_" }) + @(
    "--pool-data-seeds",
    "--max-samples",    $MaxSamples,
    "--stride",         $Stride,
    "--distant-stride", $DistantStride,
    "--selection",      $Selection
)
if ($FullTest) {
    $ScriptArgs += "--full-test"
}

$TsTag = $TrainingSeeds -join "_"
Write-Host "Using Python:     $Python"
Write-Host "Project root:     $ProjectRoot"
Write-Host "Output dir:       $OutDir"
Write-Host "Training seeds:   $($TrainingSeeds -join ', ')"
Write-Host "Max samples/split: $MaxSamples (total ~$($MaxSamples * 3) pooled)"
Write-Host ""
Write-Host "Outputs written to:"
Write-Host "  $OutDir\ds101_ts$TsTag\"
Write-Host "  $OutDir\ds202_ts$TsTag\"
Write-Host "  $OutDir\ds303_ts$TsTag\"
Write-Host "  $OutDir\pooled_ts$TsTag\"

& $Python @ScriptArgs
