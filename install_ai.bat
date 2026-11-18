@echo off
rem Compatibility entry point: install the complete application environment.
call "%~dp0setup.bat" %*
exit /b %errorlevel%
