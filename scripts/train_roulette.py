"""Train and evaluate the whole connectome for the roulette game."""
import argparse
import json
import math
from pathlib import Path

from flybrain.roulette_brain import RouletteBrain
from flybrain.roulette_training import train
from flybrain.roulette_reports import save_training_report

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--backend", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--evaluation-games", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.evaluation_games < 2:
        parser.error("steps, max-seconds and evaluation-games must be positive (games >= 2)")
    brain = RouletteBrain(ROOT / "data", seed=42, backend=args.backend)
    if (brain.neuron_count, brain.edge_count) != (138639, 15091983):
        raise ValueError("Expected the entire FlyWire v783 graph")
    if args.checkpoint:
        brain.load(args.checkpoint)
    report = train(brain, args.output_dir, steps=args.steps, seed=args.seed,
                   max_seconds=args.max_seconds, evaluation_games=args.evaluation_games,
                   progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    brain.save(ROOT / "outputs" / "roulette-latest.npz")
    save_training_report(ROOT / "outputs" / "roulette-latest.json", report)
