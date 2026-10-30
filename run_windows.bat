@echo off
REM Double-click to start the app. Close this window or press Ctrl+C to stop.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_app.py
pause
