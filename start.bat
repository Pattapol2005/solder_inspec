@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call "%~dp0setup.bat"
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -c "import importlib.util; assert all(importlib.util.find_spec(m) for m in ('tkinter','cv2','cv2_enumerate_cameras','PIL','psutil','send2trash','rapidocr','rapidocr_onnxruntime','torch','torchvision','ultralytics'))" >nul 2>&1
if errorlevel 1 (
    call "%~dp0setup.bat"
    if errorlevel 1 exit /b 1
)
if not exist "best_inspec.pt" (
    call "%~dp0setup.bat" -ModelsOnly
    if errorlevel 1 exit /b 1
)
if not exist "best_locator.pt" (
    call "%~dp0setup.bat" -ModelsOnly
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    pause
    exit /b 1
)
