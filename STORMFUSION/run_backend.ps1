# Launch FastAPI Backend
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

$venvPython = "$PSScriptRoot\..\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
}
