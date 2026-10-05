@echo off
rem One-time setup: Python virtual environment + PyTorch with CUDA for RTX 50-series GPUs.
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set PYTHONUTF8=1

set "PY="
for %%V in (3.12 3.13 3.11) do if not defined PY (py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V")
if not defined PY (python -c "import sys" >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  goto :fail
)
echo Using %PY%
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"
%VPY% -m pip install --upgrade pip || goto :fail

rem RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8 or newer PyTorch build.
%VPY% paper_study\check_gpu.py >nul 2>nul && goto :torch_ok
for %%C in (cu130 cu129 cu128) do (
  echo.
  echo === Trying the PyTorch CUDA build %%C
  %VPY% -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/%%C
  %VPY% paper_study\check_gpu.py && goto :torch_ok
)
echo No PyTorch CUDA build worked on this GPU. Update the NVIDIA driver and try again.
goto :fail

:torch_ok
echo.
echo === Installing the other packages
%VPY% -m pip install -r paper_study\requirements-windows.txt || %VPY% -m pip install numpy pygame-ce tiktoken regex pyarrow matplotlib || goto :fail
echo.
echo === GPU benchmark
%VPY% paper_study\check_gpu.py --bench || goto :fail
echo.
echo Setup finished. Next: double-click paper_study\run_windows.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. See README_WINDOWS.md, section Troubleshooting.
pause
exit /b 1
