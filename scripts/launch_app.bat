@echo off
echo ========================================
echo    INDICSIGNAI - Launching Application
echo ========================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Working directory: %CD%
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing/updating requirements...
pip install -r requirements.txt

REM Run the app
echo.
echo Starting Flask server...
echo.
echo Open browser at: http://localhost:5000
echo.
python app.py

pause
