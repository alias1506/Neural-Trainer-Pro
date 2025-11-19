@echo off
echo ========================================
echo Python Environment Migration to 3.11
echo ========================================
echo.

echo Step 1: Backing up current requirements...
.venv\Scripts\pip.exe freeze > requirements_backup.txt
echo   - Requirements backed up to requirements_backup.txt
echo.

echo Step 2: Removing old virtual environment...
if exist .venv (
    rmdir /s /q .venv
    echo   - Old .venv removed
) else (
    echo   - No existing .venv found
)
echo.

echo Step 3: Creating new virtual environment with Python 3.11...
py -3.11 -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Failed to create virtual environment
    echo   Make sure Python 3.11 is installed and in PATH
    pause
    exit /b 1
)
echo   - New .venv created with Python 3.11
echo.

echo Step 4: Verifying Python version...
.venv\Scripts\python.exe --version
echo.

echo Step 5: Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
echo.

echo Step 6: Installing core PyTorch dependencies...
echo   Installing PyTorch with CUDA 11.8 support...
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.

echo Step 7: Installing other dependencies...
.venv\Scripts\pip.exe install pillow numpy
echo.

echo Step 8: Installing model conversion dependencies...
echo   - Installing ONNX...
.venv\Scripts\pip.exe install onnx
echo   - Installing CoreMLTools...
.venv\Scripts\pip.exe install coremltools
echo   - Installing TensorFlow and ONNX-TF...
.venv\Scripts\pip.exe install tensorflow onnx-tf
echo.

echo Step 9: Verifying GPU detection...
.venv\Scripts\python.exe server/check_gpu.py
echo.

echo ========================================
echo Migration Complete!
echo ========================================
echo.
echo Your environment is now ready with:
echo   - Python 3.11
echo   - PyTorch with CUDA support
echo   - All model conversion tools
echo.
echo Next steps:
echo   1. Restart your terminals
echo   2. Test GPU detection
echo   3. Try model conversions
echo.
pause
