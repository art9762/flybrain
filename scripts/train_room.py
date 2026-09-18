"""Run a bounded, autonomous full-connectome training trajectory in the room.

Example from the project root:
    PYTHONPATH=src python3 scripts/train_room.py --backend mps --steps 2000

Only raw eye pixels reach the controller. Rewards come from World.step; this
runner adds no navigation reward, manual reward, action demonstrations, or
curriculum. The recorded World configuration includes any invalid-action
penalty. Output is a training record, not evidence of learned activities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from flybrain.brain import Brain
from flybrain.world import ACTIONS, World


ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    files = {"report": output / "report.json", "actions": output / "actions.json",
             "world": output / "world-final.json", "checkpoint": output / "room-brain.npz"}
    existing = [path for path in files.values() if path.exists()]
    if existing:
        raise ValueError("Training outputs already exist; choose a new --output-dir: "
                         + ", ".join(str(path) for path in existing))
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print(f"Loading the full connectome on requested backend {args.backend}…", flush=True)
    brain = Brain(args.data_dir, seed=args.seed, backend=args.backend)
    if brain.neuron_count != 138_639 or brain.edge_count != 15_091_983:
        raise ValueError("This runner requires all 138639 FlyWire neurons and 15091983 edges")
    if args.checkpoint is not None:
        brain.load(args.checkpoint)
        brain.reset_activity()
    # A new World is explicit even when weights are warm-started. There is no
    # claim to continue a saved physical state or prior roulette trajectory.
    world = World(seed=args.seed)
    initial_config = world.snapshot()["config"]
    initial_brain = brain.observe_status()
    initial_weights = brain.log_multipliers.copy()
    print(f"Loaded {brain.neuron_count:,} neurons / {brain.edge_count:,} edges; "
          f"selected={brain.operator.status['backend']}; starting at ({world.x}, {world.z}).", flush=True)
    training_started = perf_counter()
    reward_values: list[float] = []
    action_indices: list[int] = []
    action_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    proximity_entries: Counter[str] = Counter()
    previously_near: set[str] = set()
    timing_ms: list[float] = []
    xs, zs = [world.x], [world.z]
    interactions: list[dict] = []
    stop_reason = "step_limit"
    for step in range(args.steps):
        # Loading counts toward the budget. A step already started completes
        # atomically; serialization/checkpoint writes occur after the cap.
        if perf_counter() - started >= args.max_seconds:
            stop_reason = "wall_time_limit"
            break
        tick = perf_counter()
        image = world.frame()
        if image.shape != (48, 64, 3) or image.dtype != np.uint8:
            raise ValueError("World observation is not a 48x64 RGB uint8 image")
        chosen = brain.act(image, learn=True)
        result = world.step(chosen)
        brain.feedback(result["reward"], learn=True,
                       action_effective=result["action_effective"],
                       outcome_ready=result["outcome_ready"])
        action_indices.append(int(chosen))
        action_counts[ACTIONS[chosen]] += 1
        event_counts[result["event"]] += 1
        reward_values.append(float(result["reward"]))
        timing_ms.append((perf_counter() - tick) * 1000)
        xs.append(world.x)
        zs.append(world.z)
        # Ground-truth positions below are measurement-only. They never enter
        # Brain.act, feedback, action selection or reward construction.
        nearby = {sid for sid, station in world.stations.items()
                  if math.hypot(world.x - station["x"], world.z - station["z"])
                  <= world.INTERACTION_RADIUS}
        proximity_entries.update(nearby - previously_near)
        previously_near = nearby
        if world.last_interaction and world.last_interaction["phase"] == "complete":
            completed = world.last_interaction
            if not interactions or interactions[-1]["ends_at"] != completed["ends_at"]:
                interactions.append(dict(completed))
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{args.steps} | reward {world.total_reward:.2f} | "
                  f"completed uses {sum(s['uses'] for s in world.stations.values())} | "
                  f"bank {world.bank} | elapsed {perf_counter() - started:.1f}s", flush=True)

    trajectory_seconds = perf_counter() - training_started
    completed_steps = len(action_indices)
    segment = min(200, completed_steps)
    final_brain = brain.observe_status()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "autonomous room training trajectory",
        "evidence_scope": "one seed, training rewards only; no frozen evaluation or acquisition claim",
        "seed": args.seed,
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "wall_cap_seconds_including_load": args.max_seconds,
        "stop_reason": stop_reason,
        "trajectory_elapsed_seconds": trajectory_seconds,
        "simulation_seconds": world.time_s,
        "backend": brain.operator.status,
        "graph": {"neurons": brain.neuron_count, "edges": brain.edge_count,
                  "checksum": brain.graph_checksum},
        "observation": {"input": "raw world RGB image only", "shape": [48, 64, 3], "dtype": "uint8",
                        "symbolic_world_state_reaches_policy": False},
        "training": {"learning": True, "human_rewards": 0,
                     "navigation_shaping": initial_config["navigation_shaping"],
                     "delayed_activity_credit": "retain effective choices across ignored scripted ticks",
                     "curriculum": None, "initial_world_config": initial_config,
                     "warm_start_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                     "warm_start_scope": "brain weights and baseline in a fresh World; not full-session resume"
                     if args.checkpoint else "fresh brain and World",
                     "additional_learning_updates": final_brain["learningUpdates"] - initial_brain["learningUpdates"],
                     "additional_frames": final_brain["frames"] - initial_brain["frames"],
                     "weight_log_multiplier_change_norm": float(np.linalg.norm(brain.log_multipliers - initial_weights)),
                     "changed_existing_weights_at_end": final_brain["changedWeights"]},
        "rewards": {"total": world.total_reward, "nonzero_steps": int(np.count_nonzero(reward_values)),
                    "segment_steps": segment, "segments_overlap": completed_steps < 2 * segment,
                    "first_segment_mean": float(np.mean(reward_values[:segment])) if segment else None,
                    "last_segment_mean": float(np.mean(reward_values[-segment:])) if segment else None,
                    "per_step": reward_values},
        "actions": {action: action_counts[action] for action in ACTIONS},
        "events": dict(event_counts),
        "stations": {sid: {"proximity_entries": proximity_entries[sid], "completed_uses": station["uses"]}
                     for sid, station in world.stations.items()},
        "completed_interactions": interactions,
        "trajectory_extents": {"x_min": min(xs), "x_max": max(xs), "z_min": min(zs), "z_max": max(zs)},
        "final_bank": world.bank,
        "timing": {"median_step_ms": float(np.median(timing_ms)) if timing_ms else None,
                   "mean_step_ms": float(np.mean(timing_ms)) if timing_ms else None},
        "files": {key: str(path) for key, path in files.items()},
        "checkpoint_scope": "brain only; world-final.json is a diagnostic snapshot, not a session loader",
        "world_replay": "Construct World(seed), apply initial_config, then World.step for each saved action index; "
                        "requires the same world implementation. No random policy replay is implied.",
        "learned_room_behaviors_demonstrated": False,
    }
    brain.save(files["checkpoint"])
    files["actions"].write_text(json.dumps({"seed": args.seed, "initial_config": initial_config,
                                          "action_names": list(ACTIONS), "action_indices": action_indices}, indent=2)
                                + "\n", encoding="utf-8")
    files["world"].write_text(json.dumps(world.snapshot(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["total_elapsed_seconds_including_save"] = perf_counter() - started
    files["report"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"steps": completed_steps, "stop_reason": stop_reason,
                      "reward": world.total_reward, "stations": report["stations"],
                      "bank": world.bank, "report": str(files["report"]),
                      "checkpoint": str(files["checkpoint"])}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--backend", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "room-training")
    parser.add_argument("--checkpoint", type=Path, help="Load brain only, then train in a new World")
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--steps and finite --max-seconds must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")
    run(args)
