# FlyBrain — Room of Habits

*English translation of the authoritative brief; original Russian: docs/archive/ru/PROJECT_BRIEF.ru.md.*

Date: 2026-09-17. Working brief based on the user's answers.

## Current project index

The project currently has two separate local prototypes. The room of habits runs on port 8765 and uses its own saves; the Roulette duel runs on port 8766 and uses `outputs/roulette-latest.npz`. The room constraints below apply only to the room and do not prohibit numeric state in Roulette. The Roulette server was verified stopped on 2026-09-17; to launch: `./run_roulette.sh`.

The last saved Roulette checkpoint is `outputs/roulette-20260916-153242-681514/brain.npz`, also copied to `outputs/roulette-latest.npz`. Report: `outputs/roulette-latest.json`. It contains 14,000 cumulative updates: the earlier 12,000 and the later 2,000. The full graph is the same: 138,639 neurons, 15,091,983 connections, 7,044 trainable existing connections. This is the verified state of the files, not a claim about biological skill.

Detailed decisions, rules, and constraints of the duel are recorded in [ROULETTE_BRIEF.md](ROULETTE_BRIEF.md). The room prototype and its further experiments are described below.

Current stage of the room: the learning mechanism and the local loop are verified, but reliable mastery of the five activities is not confirmed. Historical overall project regression: 155 tests passed, one GPU test was skipped in the restricted process, then 5/5 GPU tests passed in a separate Metal process; the 11 existing JavaScript panel tests also passed. These results were not re-run for this entry.

## Confirmed concept

A stylized fly lives in a 3D room with five activities: watching reels on a virtual phone, smoking a hookah, drinking whiskey from a bottle, pressing confirm in a fictional Claude Code, and playing a virtual roulette. All apps, content, and chips are currently fictional. The user watches, tunes the rewards, and manually reinforces recent decisions.

The fly learns to choose an activity and to approach an object. Grasping, sipping, puffing, viewing, and pressing are canned movements in the first version. Training the motor control of individual joints is not part of the first stage.

## User decisions

| Question | Confirmed answer |
|---|---|
| Appearance | Stylized fly in a 3D room, interacts with its legs |
| Connections | Everything is fictional for now: no accounts, real commands, or bets |
| Rewards | Automatic rules and manual "good/bad" |
| First thing to learn | Choosing activities and approaching objects; gestures are canned |
| Role of the connectome | Decisions go through the real graph; learning changes part of its connections |
| Vision | Image from the eyes from the very start; no object names/coordinates passed to the controller |
| Scale | All 138,639 neurons right away; slow learning is acceptable |
| Needs | Constant reward and a satiation mode, switchable by the user |
| Roulette | Finite bank; after a loss the user manually refills the bank |
| Speed | GPU on this Mac, Apple Metal; measure the real speedup |

## First implementation rules

The world is confined to a 12×12 room in conventional units. The step time is 0.25 seconds. Eight actions: forward, left, right, interact, bet on red, bet on black, swipe, wait. Interacting requires reachability and aiming at an object. During a canned gesture, movement is blocked.

| Activity | What earns reward | Initial values |
|---|---|---|
| Reels | A completed video view | 4 seconds, +1; a swipe restarts the view |
| Hookah | A completed puff | 2.5 seconds, +1; pause between uses |
| Whiskey | A completed sip from the bottle | 2 seconds, +1; pause between uses |
| Claude Code | Confirming a new fictional request | 0.5 seconds, +1; a new request after 5 seconds |
| Roulette | Net outcome of a bet | 100 initial chips, bet 5, red/black, 0–36 |

The numeric values are engineering starting settings. Rewards are tunable. In roulette the positive multiplier preserves the sign of the outcome: a win is positive, a loss is negative. The bet is deducted once; on a win 10 chips are returned, that is, a net gain of +5. Zero loses both bets. Chips added by the user give the agent no reward. The future outcome is not visible to the agent.

In satiation mode the pleasure from repeats decreases and recovers over time. This is a specified environment mechanic; it does not prove biological addiction. Roulette losses keep their negative value even under satiation. Behavior variety is not guaranteed: constant rewards may lead to choosing the most profitable activity, and a roulette with negative expectation may stop attracting the fly.

## Vision and rendering

The controller receives only a 64×48 RGB array from a perspective camera at height 0.75 with a 72° field of view. The geometric renderer accounts for walls, floor, objects, gaze direction, distance, and occlusion. The phone shows procedurally moving shapes. Busy, availability, and bank indicators are also drawn in the world and visible through the camera.

For the human, the room is drawn in Three.js with more detailed objects and the fly. The small "Her eyes" window shows exactly the input pixels. It is simplified geometry of the same world; it is not a downscaled screenshot of the pretty exterior. Names, coordinates, and rewards in the panel are used only for observation and environment rules.

Pixel input is not proven recognition: recognition and behavior quality are checked separately. No off-the-shelf object recognizer or decision language model is used.

## Brain and plasticity

All 138,639 neurons and 15,091,983 aggregated directed connections from the v783 snapshot are loaded, without thresholding weak connections. Each decision step includes three passes over the full graph.

A custom continuous-activity model (rate model) is used. The initial weights are transformed through the square root of the synapse count, preservation of the predicted sign, and normalization of incoming connections. In version 2 the neutral activity is 0.5, and connections transmit the deviation `2 × activity − 1`; this retains visual differences that were lost to averaging in the early version. This is artificial computational dynamics, not literal biological spike rates; biophysical parameters cannot be recovered from a single connectome.

A fixed encoder extracts brightness, contrast, and color differences from the pixels. A random sparse projection feeds them into the neurons. Eight artificially chosen motor populations determine the action probabilities; direct visual input to them is disabled. These interfaces are not presented as anatomically proven sensory/motor circuits of the fly.

At seed=42, 7,044 existing internal connections entering 64 motor neurons are trained. All other connections participate in the dynamics but do not change. Training: the local derivative of the action probability at the last step of the recurrent dynamics, an eligibility trace, and the reward minus a scalar mean. This is approximate learning with limited history accounting, not a full gradient through the entire temporal trajectory. Signs are not flipped; strength multipliers are bounded to 0.1–3.

Manual reward is applied to the recent trace of actions. It is logged separately. In evaluation mode the weights and trainable states are frozen; neural activity keeps changing.

## CPU and GPU

The full sparse graph is computed on CPU via SciPy CSR or on GPU via Metal gather/scatter. The 138,639×138,639 matrix is never expanded to dense. The GPU stores connection indices and weights; plasticity updates the required positions on the device.

Auto-selection first compares accuracy and pass time on CPU/GPU. Metal can be selected explicitly before loading the brain. The local server runs outside the restricted sandbox, because the sandbox can hide Metal availability. Internet access is not required while running; Three.js is stored locally.

Initial measurements on M4 Pro: about 6 ms per full pass on GPU versus 10.7 ms on CPU. A full brain step with connection updates after optimization: about 24.8 ms on GPU versus 35.8 ms on CPU. World time, rendering, the UI, and device thermal state affect the end-to-end speed; these numbers do not mean the same speedup for the whole program.

## Checks and results

1. Full graph, known checksums, no implicit subgraph.
2. GPU/CPU give close answers; new weights are actually used by both modes.
3. Changed pixels change the internal state / action distribution; reward changes existing synapses; the frozen model does not change them.
4. Saving restores neural state, weights, traces, counters, and the random generator. The button saves the brain; the world is not part of that file.
5. Skills remain to be evaluated across new starts and images, equal budget, and several seeds. Measure reward per play-minute, approach success, completed actions, and the bank separately.
6. The contribution of real topology remains to be compared against a frozen model, a random policy, a permuted connectome, and a small alternative network.

Full GPU-loop checks: at seed=42 — 5 world steps, median 33.48 ms; at seed=43 — 40 steps, median 26.77 ms. In these short runs the fly had not yet received automatic rewards. Manual reward changed, respectively, 7,042 of 7,044 and 5,224 of 5,224 selected existing connections; the frozen evaluation did not change them; saving restored randomness and actions. The number of selected connections depends on the initial choice of motor neurons.

A separate artificial test, "two images → two correct actions", uses the full brain and 800 trials. The early version raised the probability of rewarded actions from 12.67% to 21.39%, but failed the control with shuffled images: it learned an action preference, not image discrimination. After centering the signal, version 2 at seed=42 raised the correct-action probability from 10.52% to 41.89%, and the difference between the correct and mixed-up image–action association grew from −3.08 to +32.95 percentage points on new noisy images. This is a limited check of simple visual learning; mastery of the five activities does not follow from it. Machine report: `outputs/runtime_validation.json`; to reproduce: `scripts/validate_runtime.py`.

An independent repeat of version 2 with seed=43 and the same 800 trials: the correct-action probability on noisy images grew from 12.79% to 54.76%, and the difference between the correct and mixed-up association grew from +0.66 to +48.64 percentage points. Both categories improved in both runs. The two initial states confirm that this simple test works, but give no estimate of learning reliability in the room. Report: `outputs/runtime_validation_seed43.json`.

Early historical check of the room: 56 tests passed, including real Metal outside the restricted environment. Python and JavaScript syntax checks also passed. In the browser, 3D rendering, start/pause, single stepping with Metal, manual reward, rule changes, and brain save/load were verified. Later checks are listed in the current project index and in the repair record dated 2026-09-14 below.

Next experiment: long training in the room with save/resume and evaluation on new starts. If rare rewards are not enough, enable shaping stages or manual reinforcement and measure their contribution separately.

The first autonomous room run has already been performed: 2,000 steps / 500 seconds of world time in 67.8 seconds, median 33.27 ms per step on Metal. Without manual reinforcement and without approach reward, the fly completed the Claude confirmation once and received +1; it did not complete the other activities, and the bank stayed at 100. This is a single exploratory trajectory, not an acquired skill. Actions, a world snapshot, the report, and the brain are saved in `outputs/room-training`; replaying the actions reproduces the final world. The run is launched via `scripts/train_room.py`.

## Sources and early stub

### Repair of repeated empty actions, 2026-09-14

In a live session over 26,522 steps, 77 completed activities were recorded (41 reel views, 8 puffs, 10 sips, 18 confirmations). This is not geometry hanging: all five objects are reachable with ordinary movements from the start. The log hid movements and showed every failure, which created the impression of looping.

The deferred-reward accounting was fixed: during a canned gesture the brain continues the full computation, but the ignored choice neither adds a new gradient nor weakens the retained trace. Completion, including zero reward, is attributed to the executed decisions. A real swipe remains a new decision. Earlier v2 brain saves are compatible.

A visible, configurable penalty for an unavailable action was added: −0.02 by default, 0 disables it. This is engineering feedback, not a biological model of punishment. Object coordinates/identifiers are not fed to the policy; auto-approach and action masking were not added. Repeats are grouped in the log, and movements and the count of completed activities are shown separately.

The "Return to center" button pauses the world, restores the initial position, and cancels an unfinished gesture. It keeps trained connections, time, rewards, object uses, and the bank; it returns only a bet that has not yet been played out. It clears short-term activity and the trace, so that a new start does not reinforce the old movement. The button itself gives no reward.

115 Python tests were verified (including Metal), the 11 panel tests, and a live return-to-center on GPU. Fixing the learning mechanism does not prove improved skills in the room; that requires a separate comparative run. Before the update, the brain and a diagnostic snapshot were saved to `outputs/stuck-repair-20260914`, and the previous manual file was kept as `previous-manual-save.npz`. The old snapshot did not contain the procedural clip number, so it was restored as 0; weights, pose, counters, rewards, and the bank were restored.

- Official release of the source connections: https://doi.org/10.5281/zenodo.10676866
- Downloaded aggregated snapshot: https://github.com/eonsystemspbc/fly-brain/tree/main/data
- `data/flywire_connectivity_v783.parquet`, SHA-256 `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347`.
- `data/flywire_neurons_v783.csv`, SHA-256 `52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4`.
- Old brief: `docs/archive/INITIAL_PROJECT_BRIEF.md` (Russian originals of this brief live in docs/archive/ru/). Its recommendations about Flygym, a subgraph, and conditional sensory input were superseded by the user's answers.
- The old LIF-like panel and CLI are kept as an early experiment, but they are not the engine of the new room.
