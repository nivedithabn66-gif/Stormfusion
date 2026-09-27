@echo off
title STORMFUSION Frontend Command Center (Port 5173)
cd /d "%~dp0frontiee"
echo ===================================================
echo Starting STORMFUSION React / Vite Frontend...
echo ===================================================
npm run dev -- --host 127.0.0.1 --port 5173
pause
