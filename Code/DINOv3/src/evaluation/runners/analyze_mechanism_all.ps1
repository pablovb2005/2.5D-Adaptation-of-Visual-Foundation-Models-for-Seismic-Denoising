param(
    [string]$ProjectRoot,
    [int]$Seed = 42,
    [int]$DataSeed = 101,
    [int]$TrainingSeed = 42,
    [int]$Stride = 5,
    [Nullable[int]]$SampleIndex = $null,
    [int[]]$Ranks = $null,
    [string[]]$QueryXY = $null,
    [int]$MaxSamples = 64,
    [ValidateSet("uniform", "top-gap")]
    [string]$Selection = "uniform",
    [switch]$SkipAttention,
    [switch]$SkipRepresentations,
    [switch]$SkipFigures,
    [switch]$NoHeadBreakdown,
    [switch]$NoSaliency
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

$EvalDir = Join-Path $DinoRoot "src\evaluation"
$MechanismRoot = Join-Path $ProjectRoot "experiments\summaries\mechanism_analysis"
$AttentionDir = Join-Path $MechanismRoot "attention_maps"
$RepresentationDir = Join-Path $MechanismRoot "representation_analysis"
$FigureDir = Join-Path $MechanismRoot "mechanism_figures"
$Metadata = Join-Path $ProjectRoot "experiments\summaries\comparison_panels\comparison_metadata.csv"

Write-Host "Using Python: $Python"
Write-Host "Project root: $ProjectRoot"
Write-Host "Mechanism root: $MechanismRoot"
Write-Host "Seed: $Seed"
Write-Host "Stride: $Stride"

if (-not $SkipAttention) {
    # New main protocol: dataset is built at --data-seed so the visualized slice is
    # held out for the matching checkpoints. Samples and query points are picked
    # automatically (mid-volume, high/low amplitude); override with -Ranks / -QueryXY.
    $AttentionScript = Join-Path $EvalDir "analyze_attention.py"
    $Args = @(
        $AttentionScript,
        "--project-root", $ProjectRoot,
        "--out-dir", $AttentionDir,
        "--data-seed", $DataSeed,
        "--training-seed", $TrainingSeed,
        "--stride", $Stride
    )
    if ($null -ne $SampleIndex) {
        $Args += @("--sample-index", [string]$SampleIndex)
    }
    if ($null -ne $Ranks) {
        $Args += @("--ranks") + ($Ranks | ForEach-Object { [string]$_ })
    }
    if ($null -ne $QueryXY) {
        $Args += @("--query-xy") + $QueryXY
    }
    if ($NoHeadBreakdown) {
        $Args += "--no-head-breakdown"
    }
    if ($NoSaliency) {
        $Args += "--no-saliency"
    }
    Write-Host "`n[1/3] Attention maps (data_seed=$DataSeed, training_seed=$TrainingSeed)"
    & $Python @Args
}

if (-not $SkipRepresentations) {
    $RepresentationScript = Join-Path $EvalDir "analyze_representations.py"
    $Args = @(
        $RepresentationScript,
        "--project-root", $ProjectRoot,
        "--out-dir", $RepresentationDir,
        "--metadata", $Metadata,
        "--seed", $Seed,
        "--stride", $Stride,
        "--max-samples", $MaxSamples,
        "--selection", $Selection
    )
    Write-Host "`n[2/3] Representation metrics"
    & $Python @Args
}

if (-not $SkipFigures) {
    $FigureScript = Join-Path $EvalDir "generate_mechanism_figures.py"
    $Args = @(
        $FigureScript,
        "--project-root", $ProjectRoot,
        "--representation-dir", $RepresentationDir,
        "--attention-dir", $AttentionDir,
        "--out-dir", $FigureDir
    )
    Write-Host "`n[3/3] Mechanism figures"
    & $Python @Args
}

Write-Host "`nDone. Outputs are under: $MechanismRoot"
