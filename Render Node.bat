@echo off
rem Runs this PC as a render node for MUSS on another PC.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment is missing. Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" render_node.py %*
pause
