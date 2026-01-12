@echo off
set "VENV_DIR=sound_isolator"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment not found at %VENV_DIR%.
    echo Please ensure the 'sound_isolator' folder exists and relies on python 3.12.
    pause
    exit /b 1
)

echo Starting Sound Isolator App...
"%PYTHON_EXE%" sound_isolator_app.py

if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
