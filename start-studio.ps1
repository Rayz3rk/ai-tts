param([int]$Port = 7861, [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw '请先运行 .\scripts\setup.ps1' }
& .venv\Scripts\python.exe -m audio_studio serve --port $Port --device $Device
exit $LASTEXITCODE
