#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

compatible_python() {
  "$1" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' >/dev/null 2>&1
}

if [ -e .venv ] || [ -L .venv ]; then
  if [ ! -x .venv/bin/python ] || ! compatible_python .venv/bin/python; then
    printf '%s\n' 'O ambiente .venv existente não funciona com Python 3.11–3.13.' \
      'Ele foi preservado. Renomeie-o e execute este instalador novamente.' >&2
    exit 1
  fi
else
  mlp_python="${PYTHON:-}"
  if [ -n "$mlp_python" ]; then
    if ! compatible_python "$mlp_python"; then
      printf '%s\n' 'O Python indicado em PYTHON não está disponível ou não é 3.11, 3.12 ou 3.13.' >&2
      exit 1
    fi
  else
    for mlp_candidate in python3.12 python3.13 python3.11 python3 python; do
      if command -v "$mlp_candidate" >/dev/null 2>&1 && compatible_python "$mlp_candidate"; then
        mlp_python="$mlp_candidate"
        break
      fi
    done
    if [ -z "$mlp_python" ]; then
      printf '%s\n' 'Instale Python 3.11, 3.12 ou 3.13 e seu módulo venv.' \
        'Para escolher o interpretador: PYTHON=/caminho/python3.12 bash instalar_linux.sh' >&2
      exit 1
    fi
  fi
  "$mlp_python" -m venv .venv
fi

.venv/bin/python --version
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
printf '\nInstalação concluída. Execute: bash iniciar_linux.sh (macOS: bash iniciar_macos.sh)\n'
