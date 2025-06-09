@echo off
SETLOCAL

:: Проверяем наличие venv
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo First create it with: python -m venv venv
    pause
    exit /b 1
)

:: Активируем venv
echo Activating virtual environment...
call venv\Scripts\activate

:: Проверяем наличие transformers и torch
echo Checking dependencies...
pip show transformers torch aiomysql >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing required packages...
    pip install torch transformers aiomysql python-dotenv
)

:: Проверяем наличие CUDA (для GPU)
where nvcc >nul 2>&1
if %ERRORLEVEL% == 0 (
    set CUDA_AVAILABLE=1
    echo CUDA support detected
) else (
    set CUDA_AVAILABLE=0
    echo Using CPU mode
)

:: Запуск vectorize.py с параметрами
echo Starting vectorization...
python bot/vectorize.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Error during vectorize.py execution
    pause
    exit /b 1
)

echo Vectorization completed successfully!
pause