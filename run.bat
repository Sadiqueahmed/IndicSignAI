@echo off
echo ========================================
echo    INDICSIGNAI - Starting System
echo ========================================
echo.

REM Stay in current directory (project root)
cd /d "%~dp0"

REM Check if virtual environment exists and activate
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist sign_env\Scripts\activate.bat (
    call sign_env\Scripts\activate.bat
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run the app using the launcher
echo.
echo Starting Flask server...
echo.
echo Open browser at: http://localhost:5000
echo.
python run.py

pause
