@echo off
title Marbl Meat QC Server
cd /d "%~dp0"
echo ===================================================
echo   Starting Marbl Meat QC Server...
echo ===================================================
python start_server.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)

