@echo off
echo ========================================
echo    INDICSIGNAI - Starting System
echo ========================================
echo.

REM Change to parent directory (project root)
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist venv (
    if not exist sign_env (
        echo Creating virtual environment...
        python -m venv venv
    )
)

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist sign_env\Scripts\activate.bat (
    call sign_env\Scripts\activate.bat
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run the app from src directory
echo.
echo Starting Flask server...
echo.
echo Open browser at: http://localhost:5000
echo.
cd src
python app.py

pause
