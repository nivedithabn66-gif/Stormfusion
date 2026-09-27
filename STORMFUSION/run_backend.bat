@echo off
title STORMFUSION Backend API (Port 8000)
cd /d "%~dp0"
echo ===================================================
echo Starting STORMFUSION FastAPI Backend Server...
echo ===================================================
"..\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
if errorlevel 1 (
    echo.
    echo Trying fallback to global python...
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
