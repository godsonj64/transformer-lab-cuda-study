@echo off
rem Runs the full study (seeds 0, 1, 2; RoPE and learned; 3,000 updates each),
rem evaluates each finished run, then packs the results into one zip.
rem Extra arguments are passed on, e.g.:  run_windows.bat --seeds 2
cd /d "%~dp0\.."
set PYTHONUTF8=1
.venv\Scripts\python.exe paper_study\run_study.py --device cuda %*
if errorlevel 1 (
  echo.
  echo Stopped or failed. Run run_windows.bat again to resume where it stopped.
  pause
  exit /b 1
)
.venv\Scripts\python.exe paper_study\pack_results.py
pause
