#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  printf 'Primeiro execute: bash instalar_linux.sh\n'
  exit 1
fi
if ! .venv/bin/python -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)'; then
  printf '%s\n' 'O ambiente .venv precisa de Python 3.11, 3.12 ou 3.13.' \
    'Ele foi preservado. Consulte README.md para recriar o ambiente.' >&2
  exit 1
fi
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python app.py "$@"
