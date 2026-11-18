@echo off
rem First-time setup, including Python when needed.
call "%~dp0setup.bat" %*
exit /b %errorlevel%
