param(
    [string]$ProjectRoot,
    [string]$OutDir,
    [switch]$NoPlots
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "experiments\summaries"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Summarizer = Join-Path $DinoRoot "src\evaluation\summarize_impeccable_runs.py"
$Args = @(
    $Summarizer,
    "--project-root", $ProjectRoot,
    "--out-dir", $OutDir
)
if ($NoPlots) {
    $Args += "--no-plots"
}

Write-Host "Using Python: $Python"
Write-Host "Project root: $ProjectRoot"
Write-Host "Output dir:   $OutDir"

& $Python @Args
