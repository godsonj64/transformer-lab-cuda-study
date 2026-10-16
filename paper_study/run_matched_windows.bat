@echo off
rem Matched-initialization runs to 6,000 updates (learned retrained from scratch, RoPE
rem continued from the original checkpoints when they still hold the optimizer state).
rem Each finished run is evaluated; at the end everything is packed into one zip.
rem Extra arguments are passed on, e.g.:  run_matched_windows.bat --seeds 0
cd /d "%~dp0\.."
set PYTHONUTF8=1
.venv\Scripts\python.exe paper_study\run_matched.py --device cuda %*
if errorlevel 1 (
  echo.
  echo Stopped or failed. Run run_matched_windows.bat again to resume where it stopped.
  pause
  exit /b 1
)
pause
