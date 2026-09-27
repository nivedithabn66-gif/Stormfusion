@echo off
title STORMFUSION Launcher
cd /d "%~dp0"
echo =========================================================
echo Launching STORMFUSION Full Stack (Backend + Frontend)...
echo =========================================================

echo Starting Backend in a separate window...
start "STORMFUSION Backend API" cmd /k "run_backend.bat"

timeout /t 2 >nul

echo Starting Frontend in a separate window...
start "STORMFUSION React Frontend" cmd /k "run_frontend.bat"

echo.
echo Both servers are starting up:
echo - Frontend: http://localhost:5173/
echo - Backend:  http://localhost:8000/docs
echo.
pause
