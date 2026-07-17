@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed or is not available in PATH.
    echo Install uv first, then double-click this file again.
    pause
    exit /b 1
)

echo Starting Snake AI...
echo The first launch will install dependencies and CPU PyTorch automatically.
uv run --extra cpu snake-play

if errorlevel 1 (
    echo.
    echo [ERROR] The game failed to start. Review the message above.
    pause
    exit /b 1
)
