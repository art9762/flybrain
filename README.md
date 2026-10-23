# FlyBrain

FlyBrain is an original experimental research simulator that runs the complete fruit-fly connectome — all 138,639 neurons and 15,091,983 synapses of the FlyWire v783 release — as the decision-making brain of a small animated fly. It is a simplified artificial activity model with engineered sensory and motor interfaces, not a full biophysical emulation of a fly.

## What this is

FlyBrain ships as two separate local web prototypes, each with its own model checkpoints:

- **The habits room** (`./run_lab.sh`, http://127.0.0.1:8765) — a 3D sandbox where a stylized fly lives in a room with five fictional activities: watching reels on a virtual phone, smoking a hookah, drinking whiskey, confirming prompts in a fictional Claude Code, and playing a virtual roulette. The fly's brain receives only a 64×48 image from its eyes and learns, through a trainable subset of real connectome edges, which actions to take and when to approach an object.
- **The roulette duel** (`./run_roulette.sh`, http://127.0.0.1:8766) — a Buckshot Roulette-inspired duel against the connectome: hidden live and blank shells, eight actions, items, and training of the full graph on the numeric game state.

Both servers bind to 127.0.0.1 only and use no external accounts. Nothing in the code makes network calls at runtime: Three.js is vendored locally, and the only download involved is the dataset itself, fetched once by a helper script before the first run (see Data below). All apps, content, chips, and money shown in the room are fictional.

Every decision runs three recurrent passes over the complete sparse graph; the 138,639 × 138,639 matrix is never materialized densely. The model is an artificial rate model on measured connectivity, not a recovered biological brain or evidence of biological reasoning.

## Requirements

- Python >= 3.11 (macOS was the development platform).
- Optional GPU acceleration on Apple Silicon via PyTorch MPS: install the `[gpu]` extra and run the server from a normal local terminal with GPU access. Everything also runs on CPU without it.
- Node.js is needed only for the two browser-panel test files; nothing else uses it.

## Quick start

```sh
git clone <this-repo> && cd <repo-dir>
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[gpu,dev]'   # drop [gpu] for a CPU-only install
```

Then fetch the FlyWire v783 dataset (96 MB download, SHA-256 verified into `data/`) and start either app:

```sh
python3 scripts/download_data.py
./run_lab.sh        # habits room   -> http://127.0.0.1:8765
./run_roulette.sh   # roulette duel -> http://127.0.0.1:8766
```

Stop either server with Ctrl-C in its terminal. In the room, press Start (or Step once) — the brain loads on first run, and acceleration is chosen before loading (automatic CPU/GPU comparison, forced Metal, or CPU). In the duel, press "Wake the fly"; if `outputs/roulette-latest.npz` exists it loads automatically.

## The habits room

A 12×12 room, 0.25-second world steps, eight actions: forward, left, right, interact, bet on red, bet on black, swipe, wait. Interaction requires reach and facing; movement is locked while a committed gesture plays out. Grasp, sip, puff, watch, and press are pre-built gestures — fine motor learning is out of scope for this stage. The external Three.js view is more detailed than the fly's actual vision geometry; the "Her eyes" window shows the real input pixels.

Key controls and semantics:

| Control | Behavior |
|---|---|
| Pause / one step / speed | Observe or run accelerated trials. Speed is bounded by compute time; the "64x" label does not guarantee 64-fold real-time acceleration. |
| Recentre | Pauses and returns to the original position **without** resetting trained weights, time, rewards, or completed activities. Cancels an unfinished gesture and refunds a bet that has not been played yet. Clears short-term activity and the recent-action trace so a new start does not reinforce old movement. |
| Learning toggle | Off freezes the weights for evaluation; neural activity keeps changing. |
| Good / bad | Manual reinforcement of recent decisions; must follow a step while learning is on. |
| Reward modes | Constant reward or saturating mode, switchable; values change via sliders and an Apply button. |
| Empty-action penalty | Default −0.02 for an unavailable action (for example, using an item out of reach); 0 disables it. No penalties during a committed gesture; the reward for completing it belongs to the action actually started, not to ignored choices. |
| Roulette bank | 100 starting chips, bet 5, red/black; zero loses to both red and black. +100 tops up the bank without giving the agent any reward. |
| Save / load brain | `outputs/brain-latest.npz`; includes weights, neural activity, traces, and the random generator. The room itself is not part of the save. Press pause before loading. |

Honest caveat: pixel input is not proven object recognition. The plasticity mechanism works and is tested, but the fly is not yet reliably trained on the five activities. Numbers and limits are tracked in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## The roulette duel

A Buckshot Roulette-style duel: player and fly each have four health points, a shuffled load contains live and blank shells, and shooting yourself with a blank retains the turn. Items include a magnifying glass, beer, cigarette, saw, handcuffs, and inverter, with reloads replenishing supplies. One duel per game; the original game's campaign, exact item distribution, multiplayer, and extra items are outside this prototype. Visuals are original local artwork.

The action schema is exactly: `shoot_opponent`, `shoot_self`, `magnifier`, `beer`, `cigarette`, `saw`, `cuffs`, `inverter`. The magnifying glass reveals the next shell only to its user: neither the fly's observation nor the browser response includes the secret queue. Inverting an unknown shell does not reveal exact counts — the controller sees the remaining public information and an inferred probability. A finite action cap ends an exceptionally long game in a draw.

The same full connectome drives every decision (three recurrent passes over all 138,639 neurons and 15,091,983 edges). A fixed encoder repeats 48 normalized numeric observations — health, public shell information, the fly's private inspection, inventories, turn effects, and legal actions — into the sparse sensory projection; eight artificial motor populations produce action probabilities, and illegal actions are masked by the rules engine. Only existing edges entering the selected motor neurons are trainable (7,044 for the current initialization); signs and topology are preserved. Weights are frozen whenever the fly plays against you.

On wake, `outputs/roulette-latest.npz` (currently a model after 14,000 cumulative updates) loads automatically and the matching `outputs/roulette-latest.json` is shown as the completed training result. Playing is blocked while loading, training, evaluating, or saving. The stop button halts updates at the nearest boundary, then finishes evaluation and saves.

The training report is optional: a missing, damaged, or mismatched JSON file no longer prevents a valid checkpoint from loading or playing. Its scores are shown only when the graph and cumulative update count match; new reports also identify the evaluated weights with a policy checksum. Older reports remain compatible through the graph/update checks. Report writes replace the JSON atomically, and a stopped run keeps its stopped status after restore. A damaged checkpoint still raises a load error and can be retried after repairing the file.

## Training headless

Room (full graph, world rewards only):

```sh
PYTHONPATH=src python3 scripts/train_room.py --backend mps --steps 2000 --max-seconds 120 --output-dir outputs/my-run
```

The output directory must be new. `--checkpoint outputs/room-training/room-brain.npz` resumes weights into a freshly created room — room state is not part of the checkpoint. The first completed run lives in `outputs/room-training`: 2,000 steps (500 seconds of world time) in 67.8 seconds, median 33.27 ms per Metal step. Without manual reinforcement or approach rewards, the fly completed a Claude Code confirmation once (+1), finished no other activity, and the bank stayed at 100. That is a single initial trajectory, not evidence of skill mastery.

Roulette:

```sh
PYTHONPATH=src python3 scripts/train_roulette.py \
  --backend mps --steps 2000 --evaluation-games 24 \
  --checkpoint outputs/roulette-latest.npz \
  --output-dir outputs/roulette-next-run
```

Use a fresh output directory; omit `--checkpoint` for a fresh initialization. Training defaults to 2,000 updates with a 600-second elapsed-time budget checked between updates; final evaluation and saving finish afterward, so total runtime can exceed the budget. The stop button stops updates at the next boundary, then evaluates and saves.

## Running tests

```sh
PYTHONPATH=src python3 -m pytest -q
node --test tests/test_room_controls.mjs tests/test_room_events.mjs
PYTHONPATH=src python3 scripts/validate_runtime.py --help
```

Recorded results (from 2026-09-16, not re-run for this text): 155 Python tests passed and one GPU test was skipped in a restricted process; all five GPU tests then passed in a real Metal process. All 11 room-panel JavaScript tests passed, and all 41 roulette tests passed. An earlier historical check of the room counted 56 passed tests, including real Metal outside the restricted environment.

An artificial "two images → two correct actions" test (full brain, 800 trials) is the clearest evidence that the plasticity mechanism learns: after signal centering, the probability of the correct action rose from 10.52% to 41.89% (seed 42) and from 12.79% to 54.76% (seed 43) at two initial states, with the correct-versus-swapped image–action association gap moving from −3.08 to +32.95 and from +0.66 to +48.64 percentage points on new noisy images. These are results for a simple visual-learning test — they are not item recognition or room mastery. Machine reports: `outputs/runtime_validation.json` and `outputs/runtime_validation_seed43.json`.

Measured full brain step: about 25 ms on GPU and 36 ms on CPU; one graph pass about 6 ms on GPU versus 11 ms on CPU. These are local measurements on an M4 Pro, not a speed guarantee on other hardware.

## Data

- Official FlyWire v783 release: https://doi.org/10.5281/zenodo.10676866
- Aggregated snapshot used here: https://github.com/eonsystemspbc/fly-brain/tree/main/data

The dataset is not committed to the repository (the parquet is ~96 MB). Download it with the helper script, which skips files that are already present and verified:

```sh
python3 scripts/download_data.py
```

The script fetches the two files from the aggregated snapshot, verifies their SHA-256 sums, and stores them under the names below in `data/`. To verify manually instead (`shasum -a 256 <file>` on macOS, `sha256sum <file>` on Linux):

- `data/flywire_connectivity_v783.parquet` — SHA-256 `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347` (upstream name `2025_Connectivity_783.parquet`)
- `data/flywire_neurons_v783.csv` — SHA-256 `52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4` (upstream name `2025_Completeness_783.csv`)

FlyWire data remains under its own CC BY 4.0 license; attribution is required when reusing it.

## Repository layout

```
src/flybrain/        core package: connectome loading, brain model, GPU backends,
                     room world/session/lab app, roulette engine/brain/training/app,
                     legacy cli.py and web.py prototypes (not used by the new room)
web/                 vanilla JS/HTML frontends for both apps; Three.js 0.170.0 is
                     vendored in web/vendor/ (license: web/vendor/THREE-LICENSE.txt)
scripts/             train_room.py, train_roulette.py, validate_runtime.py,
                     repair_session.py, download_data.py
tests/               pytest suites plus the two node panel tests
data/                FlyWire v783 snapshot (see Data above)
outputs/             checkpoints, reports, and run directories (gitignored)
docs/                archived briefs and the original Russian README (docs/archive/ru/)
```

## Documentation

- [PROJECT_BRIEF.md](PROJECT_BRIEF.md) — habits room: rules, learning mechanism, verification status, and limits. Authoritative.
- [ROULETTE_BRIEF.md](ROULETTE_BRIEF.md) — duel: rules, training pipeline, measured results. Authoritative.
- The original Russian README and briefs live in `docs/archive/ru/`.

## License

The code is released under the MIT license (see the LICENSE file). FlyWire data stays under its own CC BY 4.0 license with attribution.

## Limitations

This is a first research version. The plasticity mechanism works and is tested, but the fly is not yet reliably trained to use the five room activities; robust room mastery is not claimed. Measured results are trajectory evidence of the learning loop on specific seeds — not skill evaluation, not object recognition, and not evidence that the biological topology itself is advantageous. One initialization and small match samples (including the roulette duel results) do not establish general strategic competence. Numbers, constraints, and next experiments are tracked in the two briefs linked above.
