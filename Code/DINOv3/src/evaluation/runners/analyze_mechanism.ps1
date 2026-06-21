param(
    [string]$ProjectRoot,
    [string]$Metadata,
    [string]$OutDir,
    [ValidateSet("ms_ssim_2d", "input_psnr", "input_ms_ssim", "input_mse")]
    [string]$StratifyMetric = "ms_ssim_2d",
    [int]$TopExamples = 20
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $Metadata) {
    $Metadata = Join-Path $ProjectRoot "experiments\summaries\comparison_panels\comparison_metadata.csv"
}
if (-not $OutDir) {
    $Folder = if ($StratifyMetric -eq "ms_ssim_2d") {
        "stratification\by_2d_ms_ssim"
    } else {
        "stratification\by_$StratifyMetric"
    }
    $OutDir = Join-Path $ProjectRoot "experiments\summaries\mechanism_analysis\$Folder"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Script = Join-Path $DinoRoot "src\evaluation\analyze_mechanism.py"
$ScriptArgs = @(
    $Script,
    "--project-root", $ProjectRoot,
    "--metadata", $Metadata,
    "--out-dir", $OutDir,
    "--stratify-metric", $StratifyMetric,
    "--top-examples", $TopExamples
)

Write-Host "Using Python: $Python"
Write-Host "Project root: $ProjectRoot"
Write-Host "Metadata:     $Metadata"
Write-Host "Stratify by:  $StratifyMetric"
Write-Host "Output dir:   $OutDir"

& $Python @ScriptArgs
