#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run: bash INSTALL_MAC.sh"
  exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python camera_mac.py "$@"
