@echo off
set "VENV_DIR=sound_isolator"

echo ==========================================
echo Setting up Sound Isolator Environment...
echo ==========================================

if exist "%VENV_DIR%" (
    echo Virtual environment '%VENV_DIR%' already exists.
) else (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

if not exist "requirements.txt" (
    echo Error: requirements.txt not found!
    pause
    exit /b 1
)

echo Installing dependencies...
echo This may take a while depending on your internet connection.
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup Complete!
echo You can now run start_app.bat to launch the app.
echo ==========================================
pause
