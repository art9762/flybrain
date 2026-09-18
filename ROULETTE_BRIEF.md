# FlyBrain Roulette

*English translation of the authoritative brief; original Russian: docs/archive/ru/ROULETTE_BRIEF.ru.md.*

Date: 2026-09-17. A separate playable browser prototype in the existing FlyBrain workspace. The roulette server is currently stopped; restart it with `./run_roulette.sh`.

## Confirmed user decisions

- A Buckshot Roulette-inspired duel against the fly connectome.
- Use the entire available brain graph, not a sampled subgraph.
- Numeric game observations are allowed; a vision-only interface is not required.
- Build a playable browser prototype with a training workflow now.

## Implementation

Run `./run_roulette.sh` and open http://127.0.0.1:8766. This is separate from the previous room on port 8765 and has separate model checkpoints. Existing room saves remain compatible with that room only. On wake, the session loads `outputs/roulette-latest.npz` automatically when present and resets transient neural activity; the matching `outputs/roulette-latest.json` is shown as the completed training result. Save writes that checkpoint; restore reloads it and resets activity. Training creates a timestamped run directory, then updates both latest files. The game remains unavailable while loading, training, evaluating, or saving.

The player and fly have four health points. A shuffled load contains live and blank shells. Shooting yourself with a blank retains the turn. The prototype includes a magnifying glass, beer, cigarette, saw, handcuffs and inverter. Reloads replenish supplies. There is one duel per game; the original game's campaign, exact item distribution, multiplayer and additional items are outside this prototype. Visuals are original local artwork; no commercial game assets are used.

The action schema is exactly: `shoot_opponent`, `shoot_self`, `magnifier`, `beer`, `cigarette`, `saw`, `cuffs`, `inverter`.

The magnifying glass reveals the next shell only to its user. Neither the fly observation nor the browser response includes the secret queue. An inversion of an unknown shell does not reveal updated exact counts; the controller receives the remaining public information and its inferred probability instead. A finite action cap ends an exceptionally long game in a draw.

## What the fly actually is

The existing FlyWire v783 graph contains 138,639 neurons and 15,091,983 aggregated directed edges. Every decision runs three recurrent passes over the complete sparse graph. The model is an artificial rate model on measured connectivity, not a recovered biological brain or evidence of biological reasoning.

A fixed encoder repeats 48 normalized numeric observations into the existing sparse sensory projection. Inputs describe health, public shell information, the fly's own private inspection, inventories, turn effects and legal actions. Eight artificial motor populations produce action probabilities. Illegal actions are masked by the rules engine. There is no trained external policy head and no heuristic override of the fly's live decisions.

Only the existing edges entering the selected motor neurons are trainable (roughly 7,000 for the current initialization). Their signs and topology are preserved; multiplicative strength is bounded. All other edges remain active but fixed. Numeric inputs, motor selections, legal masking and the learning rule are engineered interfaces, not anatomical claims.

## Training

1. Collect reachable states from seeded simulated games. The teacher sees only the acting player's allowed information.
2. Start with shooting decisions without items.
3. Add demonstrations with items, including balanced examples of rarer teacher actions.
4. Use the last 10% of updates for reward-based play against a mixture of random and heuristic opponents. Damage has a small shaping reward; winning/losing adds +1/-1.
5. Freeze weights and evaluate on separate game seeds. Save the checkpoint and JSON report.

The teacher is a simple strategy, not an optimal solver. Supervised updates minimize cross entropy using a local derivative through the last recurrent step. Reward updates use the existing eligibility mechanism. Both modify existing connectome edges only.

Training defaults to 2,000 updates and a 600-second elapsed-time budget checked between updates. Sample preparation and initial evaluation count toward elapsed time but cannot be interrupted by this check. Final evaluation and saving finish afterward, so total runtime can exceed the budget. The stop button stops updates at the next boundary, then finishes evaluation and saves progress. Playing against the user always freezes learned weights.

The interface reports agreement with the teacher before/after training, alongside a separate evaluation report. Agreement is not a win rate. Reports include frozen matches against random and heuristic opponents, with alternating starting players and binomial confidence intervals. One initialization and a small match sample do not establish general strategic competence or an advantage from biological topology.

## Files and reproduction

- Browser app: `flybrain.roulette_app`.
- Game engine: `src/flybrain/roulette.py`.
- Numeric connectome adapter: `src/flybrain/roulette_brain.py`.
- Training/evaluation: `src/flybrain/roulette_training.py`.
- Latest playable checkpoint: `outputs/roulette-latest.npz`.
- Latest training report: `outputs/roulette-latest.json`.
- Individual training runs: new `outputs/roulette-*` directories with their own `brain.npz` and `report.json`.

```sh
PYTHONPATH=src python3 scripts/train_roulette.py \
  --backend mps --steps 2000 --evaluation-games 24 \
  --checkpoint outputs/roulette-latest.npz \
  --output-dir outputs/roulette-next-run
```

Use a fresh output directory. Omit `--checkpoint` for a fresh graph initialization; include it to continue from saved weights. The CLI also updates the latest checkpoint/report, so choose the output directory deliberately. Apple Metal requires a local process with GPU access; automatic mode checks both CPU and Metal where available. CPU is supported without a GPU.

## Verification status

Full-connectome Metal runs completed on this Mac:

| Run | Additional updates | Time including evaluation | Frozen wins vs random | Frozen wins vs heuristic | Held-out teacher agreement |
|---|---:|---:|---:|---:|---:|
| Initial | 2,000 | 47.7 s | 9/24 before, 12/24 after | 0/24 before, 3/24 after | 28.1% → 44.5% |
| Continued from initial weights | 10,000 | 212.0 s | 31/64 before, 55/64 after | 5/64 before, 14/64 after | 55.5% → 61.7% |
| Later saved continuation; current latest | 2,000 | 51.9 s | 20/24 before, 21/24 after | 7/24 before, 4/24 after | 67.2% → 68.0% |

The second run uses different training and evaluation seeds. Its before/after comparison uses the same 64 game seeds per opponent and 128 held-out decision situations. The first two rows use different sample sets and must not be treated as the same evaluation; the third reuses the first run's seed family. All three runs use graph initialization seed 42 and exactly 7,044 trainable existing edges. The earlier continuation reached 12,000 total updates. A later saved run added 2,000 updates, so the current latest checkpoint has 14,000 total learning updates (metadata and report agree; graph checksum is unchanged). In that latest 24-game evaluation it won 21/24 against random and 4/24 against the heuristic; held-out teacher agreement was 67.2% → 68.0%. The 12,000-update run's 95% Wilson intervals were 75.4–92.4% against random and 13.5–33.4% against the heuristic.

The controller remains imperfect. On the second run's held-out examples, average teacher probability for self-shooting fell from 41.7% to 21.7% while opponent-shooting improved. No beer teacher examples occurred in that small held-out decision set, so that set does not measure beer competence. The later 14,000-update run is not a clean independent improvement claim: it reused the first run's seed/sample family, and its heuristic score was 4/24 versus 7/24 before that continuation; held-out cross-entropy also rose slightly. The stronger random score does not mean every rule has been mastered. Raw per-action and game results are retained in `outputs/roulette-first-run/report.json`, `outputs/roulette-second-run/report.json`, and the later timestamped report.

Automated regression suite: 155 passed and one GPU test skipped in the restricted process; the five GPU tests then all passed in a real Metal process. All 11 existing room-panel JavaScript tests passed. After the final integration edits, all 41 roulette tests passed. These are recorded validations from 2026-09-16, not a rerun on 2026-09-17. Additional browser verification follows below.

Live browser checks completed: saved full graph loaded on MPS; saw use and double damage; autonomous fly turns and item use; human magnifier inspection; self-shot blank retaining the turn; saving; training from the UI and stopping it cleanly. The stop test finished at 68 updates, evaluated and saved its own archived run. A later saved run is now installed as the latest checkpoint: `outputs/roulette-20260916-153242-681514/brain.npz`, with 14,000 cumulative updates. The checkpoint stores the fly model; launching or starting a new game creates a fresh duel state. Responsive-layout behavior was not checked. The UI smoke-training checkpoint remains in its timestamped directory. The server was checked stopped on 2026-09-17.
