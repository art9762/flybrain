"""Bounded distillation and reinforcement through existing connectome edges."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .roulette import ACTIONS, RouletteGame, teacher_action


def mask_for(game, player=None):
    mask = np.zeros(8, dtype=bool)
    mask[game.legal_actions(player)] = True
    return mask


def samples(seed, count=512, items=True):
    """Collect demonstrations using only the acting player's information."""
    rng = np.random.default_rng(seed)
    result = []
    game = RouletteGame(seed=int(rng.integers(2**31)), items=items)
    while len(result) < count:
        if game.winner is not None:
            game = RouletteGame(seed=int(rng.integers(2**31)), items=items)
        actor = game.turn
        target = teacher_action(game, actor)
        result.append((game.observation(actor).copy(), mask_for(game, actor), target))
        # Mix competent and exploratory trajectories to expose imperfect states.
        action = target if rng.random() < .65 else int(rng.choice(game.legal_actions()))
        game.step(action)
    return result


def agreement(brain, dataset):
    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")
    correct = probability = loss = 0.0
    per_action = defaultdict(lambda: [0, 0.0])
    for observation, legal, target in dataset:
        brain.reset_activity()
        brain.act(observation, learn=False, legal_mask=legal)
        p = brain.action_probabilities
        probability += float(p[target])
        correct += int(np.argmax(p) == target)
        loss -= float(np.log(max(p[target], 1e-12)))
        per_action[ACTIONS[target]][0] += 1
        per_action[ACTIONS[target]][1] += float(p[target])
        brain.feedback(0, learn=False)
    return {"samples": len(dataset), "argmax_agreement": correct / len(dataset),
            "teacher_probability": probability / len(dataset), "cross_entropy": loss / len(dataset),
            "by_action": {a: {"samples": n, "teacher_probability": p/n}
                          for a, (n, p) in per_action.items()}}


def evaluate(brain, *, seed=92001, games=24):
    """Frozen matches with alternating first move; fly choices have no teacher."""
    if games < 1:
        raise ValueError("Evaluation needs at least one game")
    weights = brain.log_multipliers.copy()
    original_rng = deepcopy(brain.rng.bit_generator.state)
    result = {}
    for opponent in ("random", "heuristic"):
        brain.rng = np.random.default_rng(seed + 31000)
        wins = draws = decisions = 0
        for i in range(games):
            game = RouletteGame(seed=seed+i, starting_player=i % 2)
            rng = np.random.default_rng(seed + 7000 + i)
            while game.winner is None:
                if game.turn == 1:
                    brain.reset_activity()
                    action = brain.act(game.observation(1), learn=False, legal_mask=mask_for(game, 1))
                    brain.feedback(0, learn=False)
                    decisions += 1
                else:
                    action = (teacher_action(game, 0) if opponent == "heuristic"
                              else int(rng.choice(game.legal_actions(0))))
                game.step(action)
            wins += game.winner == 1
            draws += game.winner == -1
        p = wins/games
        z = 1.96
        middle = (p + z*z/(2*games))/(1+z*z/games)
        radius = z*np.sqrt(p*(1-p)/games + z*z/(4*games*games))/(1+z*z/games)
        result[opponent] = {"games": games, "wins": wins, "draws": draws,
                            "losses": games-wins-draws, "win_rate": p,
                            "win_rate_95pct_wilson": [middle-radius, middle+radius],
                            "fly_decisions": decisions}
    np.testing.assert_array_equal(weights, brain.log_multipliers)
    brain.rng.bit_generator.state = original_rng
    brain.reset_activity()
    return result


def train(brain, output_dir: Path, *, steps=2000, seed=6100, max_seconds=600,
          progress=lambda value: None, stopped=lambda: False, evaluation_games=24):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    initial = brain.log_multipliers.copy()
    updates_before = brain.learning_updates
    rng = np.random.default_rng(seed)
    basic = samples(seed, 384, items=False)
    advanced = samples(seed+1, 1024, items=True)
    heldout = samples(seed+100000, 128, items=True)
    buckets = defaultdict(list)
    for sample in advanced:
        buckets[sample[2]].append(sample)
    progress({"state": "evaluating", "steps": 0, "target_steps": steps,
              "message": "Проверяем исходную муху"})
    before = agreement(brain, heldout)
    before_matches = evaluate(brain, seed=seed+200000, games=evaluation_games)
    progress({"state": "training", "teacher_agreement_before": before["argmax_agreement"],
              "evaluation_before": before_matches, "message": "Учимся на игровых ситуациях"})
    completed = episodes = 0
    loss = 0.0
    stop_reason = "step_limit"
    demonstration_steps = max(1, int(steps*.9))
    game = None
    for index in range(steps):
        if stopped() or time.perf_counter()-start > max_seconds:
            stop_reason = "user_stop" if stopped() else "time_limit"
            break
        if index < demonstration_steps:
            if index < steps*.2:
                dataset = basic
            else:
                # Half balanced classes, half naturally distributed examples.
                dataset = (buckets[int(rng.choice(list(buckets)))] if rng.random() < .5 else advanced)
            observation, legal, target = dataset[int(rng.integers(len(dataset)))]
            distribution = np.eye(8, dtype=np.float64)[target]
            metrics = brain.teach(observation, distribution, legal_mask=legal)
            loss = float(metrics["loss"])
        else:
            if game is None or game.winner is not None:
                game = RouletteGame(seed=int(rng.integers(2**31)), starting_player=1)
                brain.reset_activity()
                episodes += 1
            # Numerical observations summarize history. Retain learning traces,
            # but start the fast neural dynamics neutral, as during evaluation.
            brain.state.fill(.5)
            hp_before = game.hp.copy()
            action = brain.act(game.observation(1), learn=True, legal_mask=mask_for(game, 1))
            game.step(action)
            while game.winner is None and game.turn == 0:
                opponent_action = (teacher_action(game, 0) if rng.random() < .5
                                   else int(rng.choice(game.legal_actions(0))))
                game.step(opponent_action)
            reward = .1*((hp_before[0]-game.hp[0])-(hp_before[1]-game.hp[1]))
            if game.winner is not None:
                reward += 1 if game.winner == 1 else (-1 if game.winner == 0 else 0)
            brain.feedback(reward, learn=True)
        completed = index+1
        if completed % 20 == 0 or completed == steps:
            progress({"steps": completed, "episodes": episodes, "loss": loss,
                      "phase": "demonstrations" if index < demonstration_steps else "reinforcement",
                      "elapsed_seconds": time.perf_counter()-start})
    progress({"state": "evaluating", "steps": completed, "message": "Проверяем замороженные веса"})
    after = agreement(brain, heldout)
    after_matches = evaluate(brain, seed=seed+200000, games=evaluation_games)
    brain.reset_activity()
    checkpoint = output_dir / "brain.npz"
    brain.save(checkpoint)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "flybrain-roulette-training-v1", "seed": seed,
        "steps": completed, "target_steps": steps, "episodes": episodes,
        "stop_reason": stop_reason, "elapsed_seconds": time.perf_counter()-start,
        "budget_scope": "Checked between training updates; evaluations and checkpoint saving complete afterward.",
        "graph": {"neurons": brain.neuron_count, "edges": brain.edge_count,
                  "checksum": brain.graph_checksum, "plastic_edges": len(brain.plastic_indices)},
        "backend": brain.operator.status, "observation": "48 visible numeric values, no secret shell sequence",
        "training": "90% imitation of a visible-information heuristic, 10% reward-modulated play",
        "teacher": "engine heuristic, not an optimal solver; never used to choose the fly's live actions",
        "evaluation": after_matches, "evaluation_before": before_matches,
        "heldout_before": before, "heldout_after": after,
        "teacher_agreement_before": before["argmax_agreement"],
        "teacher_agreement_after": after["argmax_agreement"],
        "weight_change_norm": float(np.linalg.norm(brain.log_multipliers-initial)),
        "learning_updates": brain.learning_updates-updates_before,
        "cumulative_learning_updates": brain.learning_updates,
        "checkpoint": str(checkpoint), "report": str(output_dir / "report.json"),
        "limitations": "One connectome initialization and small held-out game sample; no biological cognition or superiority claim.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    progress({**report, "state": "stopped" if stop_reason == "user_stop" else "complete",
              "message": "Обучение и проверка завершены"})
    return report
