@echo off
rem One-time setup: creates .venv and installs PyTorch (CUDA 12.8), TensorRT, PySide6 and the other dependencies.
cd /d "%~dp0"
python -m venv .venv || goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 || goto :error
".venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || (
    echo.
    echo WARNING: PyTorch cannot see a CUDA GPU. Check that the NVIDIA driver is installed.
)
echo.
echo Setup complete. Start the app with "MUSS.bat".
pause
exit /b 0
:error
echo Setup failed.
pause
exit /b 1
