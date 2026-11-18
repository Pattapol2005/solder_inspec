@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" echo Setup failed. See the message above and logs\setup.
pause
exit /b %setup_result%
