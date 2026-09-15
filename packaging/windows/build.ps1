[CmdletBinding()]
param(
    [string]$BuildPython = "",
    [switch]$CheckEnvironment,
    [int]$QaTimeout = 300
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
if ([string]::IsNullOrWhiteSpace($BuildPython)) {
    $BuildPython = Join-Path $projectRoot "tmp/build-venv/Scripts/python.exe"
}
if (-not (Test-Path -LiteralPath $BuildPython -PathType Leaf)) {
    throw "Build Python does not exist: $BuildPython. Wait for isolated environment provisioning."
}
$pythonPath = (Resolve-Path -LiteralPath $BuildPython).Path
$buildArguments = @("-I", "-u", (Join-Path $PSScriptRoot "bootstrap.py"), "--qa-timeout", "$QaTimeout")
if ($CheckEnvironment) { $buildArguments += "--check-env" }
# No activation, pip mutation, base-interpreter fallback, or concurrent provisioning.
& $pythonPath @buildArguments
if ($LASTEXITCODE -ne 0) {
    throw "OptiProp portable build/check failed with exit code $LASTEXITCODE"
}
