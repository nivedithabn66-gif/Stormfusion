# Launch React/Vite Frontend
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location "$PSScriptRoot\frontiee"

npm run dev -- --host 127.0.0.1 --port 5173
