param(
    [string]$ProjectRoot,
    [string]$OutDir
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DinoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $DinoRoot "..\..")).Path
}
if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "experiments\summaries\mechanism_analysis\trained_controls"
}

$VenvPython = Join-Path $DinoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Summarizer = Join-Path $DinoRoot "src\evaluation\summarize_mechanism_controls.py"

Write-Host "Using Python: $Python"
Write-Host "Project root: $ProjectRoot"
Write-Host "Output dir:   $OutDir"
Write-Host ""
Write-Host "Checking mechanism control runs and generating comparison table..."

& $Python $Summarizer --project-root $ProjectRoot --out-dir $OutDir

Write-Host ""
Write-Host "Done. Output in: $OutDir"
Write-Host ""
Write-Host "To also run full-test counterfactuals (removes 64-sample caveat):"
Write-Host "  cd $DinoRoot\src"
Write-Host "  $Python evaluation/analyze_context_counterfactuals.py --full-test"
