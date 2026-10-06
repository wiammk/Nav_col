param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localDependencies = Join-Path $projectRoot '.codex_deps'
$pythonCandidates = @(
    $env:NAVCOL_PYTHON,
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $projectRoot 'venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }

$runtimePython = $pythonCandidates | Select-Object -First 1
if (-not $runtimePython) {
    $pathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPython) {
        $runtimePython = $pathPython.Source
    }
}

if (-not $runtimePython) {
    throw 'No Python interpreter found. Set NAVCOL_PYTHON or create .venv\Scripts\python.exe.'
}

$pythonPathEntries = @($projectRoot)
if (Test-Path -LiteralPath $localDependencies -PathType Container) {
    $pythonPathEntries = @($localDependencies) + $pythonPathEntries
}
if ($env:PYTHONPATH) {
    $pythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

& $runtimePython @PythonArgs
exit $LASTEXITCODE
