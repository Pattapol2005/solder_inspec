@echo off
rem Force the official NVIDIA CUDA 12.8 PyTorch package.
call "%~dp0setup.bat" -Acceleration CUDA %*
exit /b %errorlevel%
