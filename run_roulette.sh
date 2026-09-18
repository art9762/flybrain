#!/bin/zsh
set -e
cd "${0:A:h}"
export PYTHONPATH="$PWD/src"
exec python3 -m uvicorn flybrain.roulette_app:app --host 127.0.0.1 --port 8766 --no-access-log
