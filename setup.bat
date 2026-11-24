@echo off
REM One-click project setup: creates venv, installs libraries, checks the data.
REM All output is saved to setup_log.txt so it can be reviewed later.
cd /d "%~dp0"
echo Setting up the project. This can take a few minutes, please wait...
call :main > setup_log.txt 2>&1
type setup_log.txt
echo.
echo Setup finished. The full log is in setup_log.txt.
pause
exit /b

:main
python --version || exit /b 1
python -m venv venv || exit /b 1
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1
git init
python -m src.check_data
exit /b 0
