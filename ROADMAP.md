# FlyBrain roadmap

Updated 2026-10-23. Priorities describe the remaining work; they are not claims of learned behavior or delivery dates. The [room brief](PROJECT_BRIEF.md) and [duel brief](ROULETTE_BRIEF.md) remain authoritative for model and game rules.

## Evidence today

- **Implemented:** separate local room and duel prototypes, full sparse-connectome decisions, bounded plasticity, checkpointing, and roulette training/evaluation. The checkpoint recovery change passed 173 Python and 11 Node tests on 2026-10-23, covering synthetic graphs and session/API behavior.
- **Historically measured:** full-graph CPU/Metal timing, simple image–action learning, one initial room trajectory, and roulette matches are recorded in the two briefs. Those experiments were not rerun during the maintenance change.
- **Unverified:** reliable mastery of all five room activities, broad roulette competence, an advantage from biological topology, and performance on other hardware. Dataset files, trained checkpoints, and raw historical reports under `outputs/` are not distributed by this Git repository.

## Priorities and completion criteria

| Priority | Remaining work | Acceptance evidence |
|---|---|---|
| P1 | Evaluate room behavior after longer training, using new starts and several seeds. | Publish frozen-policy results with equal budgets: reward per simulated minute, approach success, completions per activity, and bank outcomes. Verify evaluation leaves weights unchanged. Compare against frozen and random policies before claiming acquired skills. See [room checks](PROJECT_BRIEF.md#checks-and-results). |
| P1 | Strengthen roulette evaluation beyond the existing initialization and small samples. | Use independent training/held-out seed families and multiple model initializations; retain per-action coverage, especially beer and self-shooting, alongside matched before/after win counts and confidence intervals. Report regressions as well as gains. See [duel verification](ROULETTE_BRIEF.md#verification-status). |
| P2 | Test whether the measured connectome topology contributes to results. | Compare the same task and compute/training budget against a permuted connectome and a small alternative network, in addition to frozen/random controls. Preserve seeds and reports; do not infer a topology benefit from the current demonstrations alone. |
| P2 | Finish browser acceptance, including responsive layouts. | Record room start/pause/step/recentre and duel play/train/stop/save/load at small and large viewports, with CPU and available Metal backends. Check error recovery through the UI; keep synthetic regression results separate from full-graph/browser evidence. Responsive duel behavior is still unverified in the brief. |
| P2 | Make reproduction and regression checks repeatable from a fresh checkout. | Automate the dataset-free Python/Node suites, and document a separate checksum-verified full-graph run with hardware, dependency versions, settings, seeds, timings, and report locations. A fresh install must not assume the historical local checkpoints exist. |

## Scope boundaries

The room receives eye pixels; the duel receives its documented numeric observation. Fine joint-motor learning, biological cognition, and the original commercial game's campaign are outside the current prototypes. Room checkpoints contain the brain, not a complete world/session snapshot; loading weights into a new room must remain explicit. New learning rules or shaping experiments should be specified and evaluated separately from repairs.
