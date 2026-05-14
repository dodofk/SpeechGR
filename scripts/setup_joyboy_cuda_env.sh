#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [[ -z "${VENV_DIR}" || "${VENV_DIR}" == "/" ]]; then
  echo "Refusing unsafe VENV_DIR=${VENV_DIR}" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="uv"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
else
  echo "uv was not found. Install uv or add it to PATH first." >&2
  exit 1
fi

echo "[setup] removing ${VENV_DIR}"
rm -rf "${VENV_DIR}"

echo "[setup] creating ${VENV_DIR} with Python ${PYTHON_VERSION}"
"${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"

echo "[setup] installing requirements with Joyboy-compatible CUDA 12.1 PyTorch"
"${UV_BIN}" pip install \
  --python "${VENV_PYTHON}" \
  --extra-index-url "${PYTORCH_INDEX_URL}" \
  -r requirements.txt \
  -c constraints/joyboy-cu121.txt

"${VENV_PYTHON}" - <<'PY'
import torch

print(f"[setup] torch={torch.__version__}")
print(f"[setup] torch.version.cuda={torch.version.cuda}")
print(f"[setup] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[setup] torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("[setup] CUDA is still unavailable after install")
print(f"[setup] selected_device={torch.cuda.get_device_name(0)}")
PY
