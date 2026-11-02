# Run the whole benchmark on this computer with one command (Windows).
# Run the same command again to resume after a stop.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Value($Name) {
    Get-Content reproducibility/toolchain.txt |
        Where-Object { $_ -match "^$Name=" } |
        ForEach-Object { $_.Substring($Name.Length + 1) }
}
function Check($Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}

# A private, pinned uv in .tools\ leaves any other uv on this computer alone.
$Uv = Join-Path $PSScriptRoot ".tools\uv.exe"
$UvVersion = Value "uv"
if (-not (Test-Path $Uv) -or ((& $Uv --version).Split(" ")[1] -ne $UvVersion)) {
    Write-Host "Installing uv $UvVersion into .tools\"
    $env:UV_UNMANAGED_INSTALL = Join-Path $PSScriptRoot ".tools"
    Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
}
$env:PATH = (Join-Path $PSScriptRoot ".tools") + ";" + $env:PATH

$Pythons = @(Value "python")
& $Uv python install @Pythons; Check "uv python install"
$Poetry = "poetry==" + (Value "poetry")
$env:POETRY_VIRTUALENVS_IN_PROJECT = "true"
$Controller = & $Uv python find --managed-python $Pythons[0]; Check "uv python find"
& $Uv tool run --from $Poetry poetry env use $Controller | Out-Null; Check "poetry env use"
& $Uv tool run --from $Poetry poetry sync --quiet; Check "poetry sync"
& .venv\Scripts\python.exe -m bench.reproduce @args
exit $LASTEXITCODE
