@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
if exist ".venv" goto :existing
if defined PYTHON goto :selected

rem Priorize uma versao compativel, mesmo quando py -3 escolher 3.14.
py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
if not errorlevel 1 (
  set "MLP_PYTHON=py -3.12"
  goto :create
)
py -3.13 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>nul
if not errorlevel 1 (
  set "MLP_PYTHON=py -3.13"
  goto :create
)
py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>nul
if not errorlevel 1 (
  set "MLP_PYTHON=py -3.11"
  goto :create
)
python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if not errorlevel 1 (
  set "MLP_PYTHON=python"
  goto :create
)
echo Instale Python 3.11, 3.12 ou 3.13. Consulte README.md.
echo Para escolher o executavel, defina PYTHON com o caminho de python.exe.
goto :failed

:selected
"%PYTHON%" -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 (
  echo O Python indicado em PYTHON nao esta disponivel ou nao e 3.11, 3.12 ou 3.13.
  goto :failed
)
set MLP_PYTHON="%PYTHON%"

:create
%MLP_PYTHON% -m venv .venv
if errorlevel 1 goto :failed
goto :install

:existing
if not exist ".venv\Scripts\python.exe" goto :incompatible
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :incompatible

:install
".venv\Scripts\python.exe" --version
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :failed
echo Instalacao concluida. Execute iniciar_windows.bat.
if not defined CI pause
exit /b 0

:incompatible
echo O ambiente .venv existente nao funciona com Python 3.11, 3.12 ou 3.13.
echo Ele foi preservado. Renomeie-o e execute este instalador novamente.
:failed
echo Falha na instalacao. Consulte a mensagem acima e o README.md.
if not defined CI pause
exit /b 1
