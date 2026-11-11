@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_CMD=py -3.12"
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if not errorlevel 1 goto python_ok
set "PYTHON_CMD=python"
python -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if not errorlevel 1 goto python_ok
echo Python 3.12 not found. Install Python 3.12 from python.org first.
goto failed
:python_ok
%PYTHON_CMD% -c "import platform; assert platform.machine().upper() in ('AMD64','X86_64'), 'This bundle requires Intel/AMD 64-bit Windows'"
if errorlevel 1 goto failed
if exist ".venv\Scripts\python.exe" goto env_ready
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto failed
:env_ready
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12), 'Use a fresh folder with Python 3.12'"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-windows.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" camera_windows.py --check
if errorlevel 1 goto failed
echo.
echo INSTALL COMPLETE. Open RUN_WINDOWS.cmd or CAMERA_0_WINDOWS.cmd.
pause
exit /b 0
:failed
echo.
echo INSTALL FAILED. Read the error above. See START_HERE.html for help.
pause
exit /b 1
