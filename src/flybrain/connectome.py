from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Connectome:
    """Compact outgoing adjacency for a directed, signed connectome."""

    neuron_ids: np.ndarray
    indptr: np.ndarray
    targets: np.ndarray
    weights: np.ndarray

    @property
    def neuron_count(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.targets.size)

    def outgoing(self, neuron_index: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = self.indptr[neuron_index : neuron_index + 2]
        return self.targets[start:stop], self.weights[start:stop]


def load_connectome(data_dir: Path, minimum_synapses: int = 5) -> Connectome:
    """Load FlyWire v783 and build a memory-efficient outgoing adjacency."""
    metadata_path = data_dir / "flywire_neurons_v783.csv"
    connectivity_path = data_dir / "flywire_connectivity_v783.parquet"
    if not metadata_path.exists() or not connectivity_path.exists():
        raise FileNotFoundError(
            "FlyWire data is missing. Expected flywire_neurons_v783.csv and "
            "flywire_connectivity_v783.parquet in the data directory."
        )

    metadata = pd.read_csv(metadata_path, usecols=["Unnamed: 0"])
    neuron_ids = metadata["Unnamed: 0"].to_numpy(dtype=np.int64, copy=True)
    frame = pd.read_parquet(
        connectivity_path,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Connectivity",
            "Excitatory",
        ],
        filters=[("Connectivity", ">=", minimum_synapses)],
    )
    frame.sort_values("Presynaptic_Index", kind="stable", inplace=True)

    sources = frame["Presynaptic_Index"].to_numpy(dtype=np.int32, copy=False)
    targets = frame["Postsynaptic_Index"].to_numpy(dtype=np.int32, copy=True)
    strengths = frame["Connectivity"].to_numpy(dtype=np.float32, copy=False)
    signs = frame["Excitatory"].to_numpy(dtype=np.float32, copy=False)
    weights = np.multiply(strengths, signs, dtype=np.float32)

    counts = np.bincount(sources, minlength=neuron_ids.size)
    indptr = np.empty(neuron_ids.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=indptr[1:])
    return Connectome(neuron_ids, indptr, targets, weights)


def choose_seed(connectome: Connectome) -> int:
    """Choose a strongly connected neuron as an interesting default stimulus."""
    degrees = np.diff(connectome.indptr)
    return int(np.argmax(degrees))


def extract_reachable_subgraph(
    connectome: Connectome,
    seed: int,
    maximum_neurons: int = 5_000,
    maximum_hops: int = 3,
) -> Connectome:
    """Breadth-first sample around a seed while preserving original neuron IDs."""
    if not 0 <= seed < connectome.neuron_count:
        raise ValueError(f"seed must be in [0, {connectome.neuron_count - 1}]")
    if maximum_neurons < 1 or maximum_hops < 0:
        raise ValueError("maximum_neurons must be positive and maximum_hops non-negative")

    selected = [seed]
    seen = {seed}
    frontier = [seed]
    for _ in range(maximum_hops):
        next_frontier: list[int] = []
        for source in frontier:
            targets, _ = connectome.outgoing(source)
            for target in targets:
                target_int = int(target)
                if target_int not in seen:
                    seen.add(target_int)
                    selected.append(target_int)
                    next_frontier.append(target_int)
                    if len(selected) >= maximum_neurons:
                        break
            if len(selected) >= maximum_neurons:
                break
        frontier = next_frontier
        if not frontier or len(selected) >= maximum_neurons:
            break

    selected_array = np.asarray(selected, dtype=np.int32)
    old_to_new = np.full(connectome.neuron_count, -1, dtype=np.int32)
    old_to_new[selected_array] = np.arange(selected_array.size, dtype=np.int32)
    sub_sources: list[np.ndarray] = []
    sub_targets: list[np.ndarray] = []
    sub_weights: list[np.ndarray] = []
    for new_source, old_source in enumerate(selected_array):
        targets, weights = connectome.outgoing(int(old_source))
        mapped = old_to_new[targets]
        keep = mapped >= 0
        if keep.any():
            sub_sources.append(np.full(int(keep.sum()), new_source, dtype=np.int32))
            sub_targets.append(mapped[keep])
            sub_weights.append(weights[keep])

    if sub_targets:
        sources = np.concatenate(sub_sources)
        targets = np.concatenate(sub_targets)
        weights = np.concatenate(sub_weights).astype(np.float32, copy=False)
    else:
        sources = np.empty(0, dtype=np.int32)
        targets = np.empty(0, dtype=np.int32)
        weights = np.empty(0, dtype=np.float32)
    counts = np.bincount(sources, minlength=selected_array.size)
    indptr = np.empty(selected_array.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=indptr[1:])
    return Connectome(connectome.neuron_ids[selected_array], indptr, targets, weights)
