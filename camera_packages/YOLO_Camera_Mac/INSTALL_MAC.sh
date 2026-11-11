#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "INSTALL FAILED. Read the error above; see START_HERE.html." >&2' ERR
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 not found. Install the macOS universal2 .pkg from python.org."
  exit 1
fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3,12)'
arch="$(python3.12 -c 'import platform; print(platform.machine())')"
case "$arch" in
  arm64) req="requirements-mac-apple.txt" ;;
  x86_64) req="requirements-mac-intel.txt" ;;
  *) echo "Unsupported architecture: $arch"; exit 1 ;;
esac
echo "Python architecture: $arch / $req"
if [ ! -x .venv/bin/python ]; then python3.12 -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use a fresh folder with Python 3.12"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "$req"
.venv/bin/python -m pip check
.venv/bin/python camera_mac.py --check
echo "INSTALL COMPLETE. Run: bash RUN_MAC.sh"
