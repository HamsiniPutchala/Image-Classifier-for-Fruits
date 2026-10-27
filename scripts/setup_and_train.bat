@echo off
setlocal
cd /d "%~dp0.."

echo ================================================
echo FruitVision AI - Dataset + CNN Training Setup
echo ================================================

where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)

echo.
echo [1/4] Creating virtual environment if needed...
if not exist .venv (
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo.
echo [2/4] Installing project dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [3/4] Downloading and preparing Apple/Banana/Orange dataset...
python scripts\prepare_dataset.py
if errorlevel 1 goto :error

echo.
echo [4/4] Training CNN and creating models\best_model.keras...
python train.py
if errorlevel 1 goto :error

echo.
echo ================================================
echo DONE! The trained model is in models\best_model.keras
echo Start the app with: streamlit run app.py
echo ================================================
pause
exit /b 0

:error
echo.
echo Setup or training failed. Read the error above.
pause
exit /b 1
