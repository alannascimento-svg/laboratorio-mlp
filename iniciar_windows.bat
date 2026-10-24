@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Primeiro execute instalar_windows.bat.
  if not defined CI pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 (
  echo O ambiente .venv precisa de Python 3.11, 3.12 ou 3.13.
  echo Ele foi preservado. Consulte README.md.
  if not defined CI pause
  exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
".venv\Scripts\python.exe" app.py %*
set "MLP_EXIT_CODE=%ERRORLEVEL%"
if not defined CI pause
exit /b %MLP_EXIT_CODE%
