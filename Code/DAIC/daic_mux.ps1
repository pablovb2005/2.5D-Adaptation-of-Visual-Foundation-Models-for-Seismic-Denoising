param(
    [Parameter(Position = 0)]
    [string]$Command = "check",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"

function Quote-Bash {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$RepoRootForWsl = $RepoRoot.Replace("\", "/")
$WslRepoRoot = (& wsl.exe wslpath -a $RepoRootForWsl).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslRepoRoot)) {
    throw "Could not translate repo path for WSL: $RepoRoot"
}

$BashParts = @(
    "cd",
    (Quote-Bash $WslRepoRoot),
    "&&",
    "bash",
    "Code/DAIC/daic_mux.sh",
    (Quote-Bash $Command)
)

foreach ($Arg in $Rest) {
    $BashParts += (Quote-Bash $Arg)
}

$BashCommand = $BashParts -join " "
& wsl.exe bash -lc $BashCommand
exit $LASTEXITCODE
