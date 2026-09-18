#!/bin/zsh
set -e
cd "${0:A:h}"
export PYTHONPATH="$PWD/src"
exec python3 -m uvicorn flybrain.lab:app --host 127.0.0.1 --port 8765 --no-access-log
