@echo off
REM Double-click this file (or run it in the VS Code terminal) to install
REM every library, train the models and run the tests on Windows.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 setup_project.py
) else (
    python setup_project.py
)
if errorlevel 1 (
    echo.
    echo Setup did not finish. If Python is missing, install Python 3.12 or 3.13 from
    echo https://www.python.org/downloads/ and tick "Add python.exe to PATH".
)
echo.
pause
