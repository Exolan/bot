@echo off
SETLOCAL

:: Проверяем, существует ли venv
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

:: Активируем venv и устанавливаем зависимости
echo Activating venv and checking dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo Error installing dependencies. Please check requirements.txt.
    pause
    exit /b 1
)

:: Запускаем бота
echo Starting bot...
python bot/main.py

if %ERRORLEVEL% neq 0 (
    echo Error starting bot.
    pause
    exit /b 1
)

pause