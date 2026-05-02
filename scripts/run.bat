@echo off
echo ========================================
echo    INDICSIGNAI - Starting System
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run the app
echo.
echo Starting Flask server...
echo.
echo Open browser at: http://localhost:5000
echo.
python app.py

pause
