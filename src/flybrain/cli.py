from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .connectome import choose_seed, extract_reachable_subgraph, load_connectome
from .simulator import SimulationConfig, simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FlyBrain Lab simulator")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--minimum-synapses", type=int, default=5)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--neurons", type=int, default=5_000, help="0 means the full graph")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--duration-ms", type=float, default=250.0)
    parser.add_argument("--timestep-ms", type=float, default=0.5)
    parser.add_argument("--gain", type=float, default=0.045)
    parser.add_argument("--stimulus", type=float, default=1.25)
    parser.add_argument("--output", type=Path, default=Path("outputs/latest.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    connectome = load_connectome(args.data_dir, args.minimum_synapses)
    original_seed = args.seed_index if args.seed_index is not None else choose_seed(connectome)
    if args.neurons:
        connectome = extract_reachable_subgraph(connectome, original_seed, args.neurons, args.hops)
        seed = 0
    else:
        seed = original_seed
    config = SimulationConfig(
        duration_ms=args.duration_ms,
        timestep_ms=args.timestep_ms,
        synaptic_gain=args.gain,
        stimulus_amplitude=args.stimulus,
        stimulus_start_ms=args.duration_ms * 0.1,
        stimulus_end_ms=args.duration_ms * 0.4,
    )
    result = simulate(connectome, seed, config).to_dict()
    result["wall_time_seconds"] = round(time.perf_counter() - started, 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "neurons": result["neuron_count"],
        "edges": result["edge_count"],
        "spikes": result["spike_count"],
        "wall_time_seconds": result["wall_time_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
