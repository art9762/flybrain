# FlyBrain

Simulator for the FlyWire v783 connectome (138,639 neurons, 15,091,983 synapses) with two separate local prototypes:

- **Room** ("habits room") — port 8765, saves to `outputs/brain-latest.npz`
- **Roulette** (Buckshot Roulette-style duel) — port 8766, saves to `outputs/roulette-latest.npz`

Details and constraints: `PROJECT_BRIEF.md` (room), `ROULETTE_BRIEF.md` (duel). These briefs are authoritative for rules and verified state — read before changing reward/learning logic. Russian originals live in `docs/archive/ru/`.

## Commands

```sh
# servers
./run_lab.sh        # room, http://127.0.0.1:8765
./run_roulette.sh   # duel, http://127.0.0.1:8766

# data (first run only; verifies SHA-256 into data/)
python3 scripts/download_data.py

# tests
PYTHONPATH=src python3 -m pytest -q
node --test tests/test_room_controls.mjs tests/test_room_events.mjs

# validate runtime / GPU backends
PYTHONPATH=src python3 scripts/validate_runtime.py --help

# headless training
PYTHONPATH=src python3 scripts/train_room.py --backend mps --steps 2000 --max-seconds 120 --output-dir outputs/<new-dir>
```

## Notes

- Python >=3.11, hatchling build, package lives in `src/flybrain`; `PYTHONPATH=src` is required for direct script runs.
- GPU is Apple Metal (MPS) on this Mac; torch is an optional extra (`[gpu]`). A separate validation run needs a normal local terminal with GPU access.
- Training output dir must be new; `--checkpoint outputs/room-training/room-brain.npz` resumes weights (room state is not part of the checkpoint).
- Web frontend is vanilla JS/HTML in `web/` (Three.js view + `vendor/`); node tests cover panel controls/events.
- Server binds 127.0.0.1 only; no external accounts or network calls.
