"""Bounded full-FlyWire runtime checks and a synthetic pixel-association probe.

Run from the repository root with ``PYTHONPATH=src python3
scripts/validate_runtime.py --backend mps``. The probe is a contextual bandit,
not evidence that navigation or room activities have been learned.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from flybrain.brain import Brain
from flybrain.world import World


def picture(label: int, rng: np.random.Generator | None = None) -> np.ndarray:
    image = np.zeros((48, 64, 3), dtype=np.float32)
    # The controller receives only pixels; label is used only by the external
    # reward evaluator, never supplied to Brain.act or encode_vision.
    if label == 0:
        image[8:40, 4:28, 0] = 1.0
    else:
        image[8:40, 36:60, 2] = 1.0
    if rng is not None:
        image = np.clip(image + rng.normal(0, 0.035, image.shape), 0, 1)
    return np.rint(image * 255).astype(np.uint8)


def evaluate(brain: Brain, noisy: bool) -> dict:
    rng = np.random.default_rng(8091)
    probabilities: list[list[float]] = [[], []]
    swapped: list[list[float]] = [[], []]
    for label in range(2):
        for _ in range(8 if noisy else 1):
            brain.reset_activity()
            brain.act(picture(label, rng if noisy else None), learn=False)
            probabilities[label].append(float(brain.action_probabilities[label + 1]))
            swapped[label].append(float(brain.action_probabilities[2 - label]))
            brain.feedback(0, learn=False)
    means = [float(np.mean(values)) for values in probabilities]
    swapped_means = [float(np.mean(values)) for values in swapped]
    return {"per_class_correct_action_probability": means,
            "per_class_swapped_action_probability": swapped_means,
            "matched_minus_swapped": float(np.mean(means) - np.mean(swapped_means)),
            "expected_reward_balanced_classes": float(np.mean(means)),
            "samples_per_class": len(probabilities[0])}


def run(args: argparse.Namespace) -> dict:
    started = perf_counter()
    brain = Brain(args.data_dir, seed=args.seed, backend=args.backend)
    world = World(seed=args.seed)
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "backend": brain.operator.status,
        "neurons": brain.neuron_count,
        "edges": brain.edge_count,
        "plastic_edges": int(brain.plastic_indices.size),
        "graph_checksum": brain.graph_checksum,
    }
    assert brain.neuron_count == 138_639 and brain.edge_count == 15_091_983
    initial_frames, initial_neural = brain.frames, brain.neural_steps
    timings = []
    rewards = []
    for _ in range(args.world_steps):
        tick = perf_counter()
        frame = world.frame()
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
        action = brain.act(frame)
        result = world.step(action)
        brain.feedback(result["reward"], action_effective=result["action_effective"],
                       outcome_ready=result["outcome_ready"])
        timings.append((perf_counter() - tick) * 1000)
        rewards.append(result["reward"])
    report["world_loop"] = {
        "steps": args.world_steps,
        "median_ms_including_camera_and_learning": float(np.median(timings)),
        "mean_ms_including_camera_and_learning": float(np.mean(timings)),
        "frames": brain.frames - initial_frames,
        "neural_steps": brain.neural_steps - initial_neural,
        "nonzero_rewards": int(np.count_nonzero(rewards)),
        "total_reward": float(np.sum(rewards)),
        "learned_room_success_demonstrated": False,
    }
    assert brain.frames - initial_frames == args.world_steps
    assert brain.neural_steps - initial_neural == args.world_steps * brain.RECURRENT_STEPS
    print("World runtime:", report["world_loop"], flush=True)

    before_weights = brain.matrix.data.copy()
    brain.reinforce(1.0)
    changed = np.flatnonzero(brain.matrix.data != before_weights)
    report["manual_reward_check"] = {
        "reward": 1.0, "changed_existing_edges": int(changed.size),
        "only_selected_edges_changed": bool(np.isin(changed, brain.plastic_indices).all()),
        "signs_preserved": bool(np.array_equal(np.sign(before_weights), np.sign(brain.matrix.data))),
        "evidence_scope": "human reward reaches existing connectome weights; not autonomous task learning",
    }
    assert changed.size > 0
    assert report["manual_reward_check"]["only_selected_edges_changed"]
    assert report["manual_reward_check"]["signs_preserved"]
    del before_weights

    weights = brain.matrix.data[brain.plastic_indices].copy()
    traces = brain.eligibility.copy()
    baseline, updates = brain.baseline, brain.learning_updates
    for _ in range(5):
        brain.act(world.frame(), learn=False)
        brain.feedback(1.0, learn=False)
    frozen = (np.array_equal(weights, brain.matrix.data[brain.plastic_indices])
              and np.array_equal(traces, brain.eligibility)
              and baseline == brain.baseline and updates == brain.learning_updates)
    report["frozen_evaluation"] = {"steps": 5, "weights_traces_critic_unchanged": frozen}
    assert frozen

    with tempfile.TemporaryDirectory(prefix="flybrain-validation-") as directory:
        checkpoint = Path(directory) / "checkpoint.npz"
        brain.save(checkpoint)
        clone = Brain(args.data_dir, seed=args.seed, backend=args.backend)
        clone.load(checkpoint)
        arrays_restored = (np.array_equal(brain.state, clone.state)
                           and np.array_equal(brain.log_multipliers, clone.log_multipliers)
                           and np.array_equal(brain.eligibility, clone.eligibility))
        rng_restored = brain.rng.bit_generator.state == clone.rng.bit_generator.state
        actions_equal = True
        max_state_error = 0.0
        for _ in range(8):
            image = world.frame()
            a, b = brain.act(image, learn=False), clone.act(image, learn=False)
            actions_equal &= a == b
            max_state_error = max(max_state_error, float(np.max(np.abs(brain.state - clone.state))))
            brain.feedback(0, learn=False)
            clone.feedback(0, learn=False)
        report["checkpoint"] = {"arrays_restored_exactly": arrays_restored,
                                "rng_restored": rng_restored,
                                "eight_replayed_actions_equal": actions_equal,
                                "max_state_error": max_state_error}
        assert arrays_restored and rng_restored and actions_equal and max_state_error < 2e-5
        del clone
    del brain

    # A fresh brain prevents the earlier human-reward test from contaminating
    # the learning result. Fixed seed 42 is a single-run smoke probe only.
    brain = Brain(args.data_dir, seed=args.seed, backend=args.backend)
    before = {"clean": evaluate(brain, False), "heldout_noisy": evaluate(brain, True)}
    probe_start = perf_counter()
    curriculum_rng = np.random.default_rng(args.seed + 17)
    earned: list[float] = []
    for step in range(args.steps):
        if perf_counter() - probe_start >= args.max_probe_seconds:
            break
        label = int(curriculum_rng.integers(0, 2))
        brain.reset_activity()
        action = brain.act(picture(label), learn=True)
        reward = float(action == label + 1)
        brain.feedback(reward, learn=True)
        earned.append(reward)
        if (step + 1) % 100 == 0:
            print(f"Synthetic visual association: {step + 1}/{args.steps}, "
                  f"recent reward={np.mean(earned[-100:]):.3f}", flush=True)
    after = {"clean": evaluate(brain, False), "heldout_noisy": evaluate(brain, True)}
    improvement = (after["heldout_noisy"]["expected_reward_balanced_classes"]
                   - before["heldout_noisy"]["expected_reward_balanced_classes"])
    both_improved = all(new > old for old, new in zip(
        before["heldout_noisy"]["per_class_correct_action_probability"],
        after["heldout_noisy"]["per_class_correct_action_probability"],
    ))
    discrimination_change = (after["heldout_noisy"]["matched_minus_swapped"]
                             - before["heldout_noisy"]["matched_minus_swapped"])
    associated = bool(after["heldout_noisy"]["matched_minus_swapped"] > 0.02
                      and discrimination_change > 0.01)
    report["synthetic_visual_association_probe"] = {
        "description": "red rectangle left -> action 1; blue rectangle right -> action 2; reward 1 for correct, 0 otherwise",
        "evidence_scope": "single-seed synthetic contextual bandit; not object recognition or learned room behavior",
        "steps_requested": args.steps,
        "steps_completed": len(earned),
        "wall_cap_seconds": args.max_probe_seconds,
        "elapsed_seconds": perf_counter() - probe_start,
        "before": before, "after": after,
        "heldout_expected_reward_improvement": improvement,
        "both_classes_improved": both_improved,
        "matched_vs_swapped_discrimination_change": discrimination_change,
        "image_specific_association_demonstrated": associated,
        "outcome": ("image_association_improved_in_this_single_seed_probe" if associated
                    else "rewarded_action_preference_increased_but_no_image_association"
                    if improvement > 0.01 and both_improved else "no_clear_convergence"),
        "training_first_100_mean_reward": float(np.mean(earned[:100])) if earned else None,
        "training_last_100_mean_reward": float(np.mean(earned[-100:])) if earned else None,
        "changed_existing_edges": brain.observe_status()["changedWeights"],
    }
    report["elapsed_seconds"] = perf_counter() - started
    report["runtime_checks_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backend", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--world-steps", type=int, default=40)
    parser.add_argument("--max-probe-seconds", type=float, default=90)
    parser.add_argument("--output", type=Path, default=Path("outputs/runtime_validation.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.world_steps < 1 or args.max_probe_seconds <= 0:
        parser.error("steps, world-steps and max-probe-seconds must be positive")
    run(args)
