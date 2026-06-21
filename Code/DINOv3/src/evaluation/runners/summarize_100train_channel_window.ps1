param(
    [string]$ProjectRoot,
    [string]$RunsRoot,
    [string]$OutDir,
    [switch]$NoPlots,
    [switch]$NoReport
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Summarizer = Join-Path $DinoRoot "src\evaluation\summarizers\summarize_100train_channel_window.py"
$Args = @(
    $Summarizer,
    "--project-root", $ProjectRoot
)
if ($RunsRoot) {
    $Args += @("--runs-root", $RunsRoot)
}
if ($OutDir) {
    $Args += @("--out-dir", $OutDir)
}
if ($NoPlots) {
    $Args += "--no-plots"
}
if ($NoReport) {
    $Args += "--no-report"
}

$SrcRoot = Join-Path $DinoRoot "src"
$env:PYTHONPATH = $SrcRoot

Write-Host "Using Python: $Python"
Write-Host "Project root: $ProjectRoot"
Write-Host "Summarizer:   $Summarizer"

& $Python @Args
