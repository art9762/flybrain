from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .connectome import Connectome


@dataclass(frozen=True)
class SimulationConfig:
    duration_ms: float = 250.0
    timestep_ms: float = 0.5
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    threshold: float = 1.0
    reset_voltage: float = 0.0
    refractory_ms: float = 2.0
    synaptic_gain: float = 0.045
    stimulus_amplitude: float = 1.25
    stimulus_start_ms: float = 20.0
    stimulus_end_ms: float = 80.0
    maximum_spikes: int = 250_000
    sample_every_steps: int = 2

    def validate(self) -> None:
        if self.duration_ms <= 0 or self.timestep_ms <= 0:
            raise ValueError("duration and timestep must be positive")
        if self.timestep_ms > self.duration_ms:
            raise ValueError("timestep cannot exceed duration")
        if self.membrane_tau_ms <= 0 or self.synapse_tau_ms <= 0:
            raise ValueError("time constants must be positive")
        if not 0 <= self.stimulus_start_ms < self.stimulus_end_ms <= self.duration_ms:
            raise ValueError("stimulus interval must lie within the simulation")


@dataclass(frozen=True)
class SimulationResult:
    config: dict
    neuron_count: int
    edge_count: int
    seed_index: int
    seed_flywire_id: int
    steps: int
    spike_count: int
    truncated: bool
    spike_times_ms: list[float]
    spike_neuron_indices: list[int]
    spike_flywire_ids: list[int]
    activity_times_ms: list[float]
    activity_counts: list[int]
    top_neurons: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def simulate(
    connectome: Connectome,
    seed_index: int,
    config: SimulationConfig,
) -> SimulationResult:
    """Run an original deterministic, discrete-time LIF-like network model.

    Connections are signed by FlyWire neurotransmitter inference and weighted by
    synapse count. This is an exploratory dynamics model, not a biophysical
    reproduction of a living fly.
    """
    config.validate()
    if not 0 <= seed_index < connectome.neuron_count:
        raise ValueError(f"seed_index must be in [0, {connectome.neuron_count - 1}]")

    neuron_count = connectome.neuron_count
    voltage = np.zeros(neuron_count, dtype=np.float32)
    synaptic_current = np.zeros(neuron_count, dtype=np.float32)
    refractory_steps = max(1, round(config.refractory_ms / config.timestep_ms))
    refractory_until = np.zeros(neuron_count, dtype=np.int32)
    membrane_decay = np.float32(np.exp(-config.timestep_ms / config.membrane_tau_ms))
    synapse_decay = np.float32(np.exp(-config.timestep_ms / config.synapse_tau_ms))
    steps = int(np.ceil(config.duration_ms / config.timestep_ms))

    spike_times: list[float] = []
    spike_neurons: list[int] = []
    activity_times: list[float] = []
    activity_counts: list[int] = []
    total_spikes = np.zeros(neuron_count, dtype=np.int32)
    truncated = False

    for step in range(steps):
        time_ms = step * config.timestep_ms
        synaptic_current *= synapse_decay
        voltage *= membrane_decay
        voltage[step < refractory_until] = config.reset_voltage
        if config.stimulus_start_ms <= time_ms < config.stimulus_end_ms:
            voltage[seed_index] += config.stimulus_amplitude

        spiking = np.flatnonzero((voltage >= config.threshold) & (step >= refractory_until))
        if spiking.size:
            total_spikes[spiking] += 1
            remaining = config.maximum_spikes - len(spike_times)
            if remaining > 0:
                recorded = spiking[:remaining]
                spike_times.extend([round(time_ms, 4)] * recorded.size)
                spike_neurons.extend(recorded.tolist())
            if spiking.size > remaining:
                truncated = True

            for source in spiking:
                start, stop = connectome.indptr[source : source + 2]
                np.add.at(
                    synaptic_current,
                    connectome.targets[start:stop],
                    connectome.weights[start:stop] * config.synaptic_gain,
                )
            voltage[spiking] = config.reset_voltage
            refractory_until[spiking] = step + refractory_steps

        voltage += synaptic_current * np.float32(config.timestep_ms / config.membrane_tau_ms)
        if step % config.sample_every_steps == 0:
            activity_times.append(round(time_ms, 4))
            activity_counts.append(int(spiking.size))

    recorded_indices = np.asarray(spike_neurons, dtype=np.int32)
    spike_ids = connectome.neuron_ids[recorded_indices].astype(np.int64).tolist()
    top_indices = np.argsort(total_spikes)[::-1]
    top_neurons = [
        {
            "index": int(index),
            "flywire_id": int(connectome.neuron_ids[index]),
            "spikes": int(total_spikes[index]),
        }
        for index in top_indices[:20]
        if total_spikes[index] > 0
    ]
    return SimulationResult(
        config=asdict(config),
        neuron_count=neuron_count,
        edge_count=connectome.edge_count,
        seed_index=seed_index,
        seed_flywire_id=int(connectome.neuron_ids[seed_index]),
        steps=steps,
        spike_count=int(total_spikes.sum()),
        truncated=truncated,
        spike_times_ms=spike_times,
        spike_neuron_indices=spike_neurons,
        spike_flywire_ids=spike_ids,
        activity_times_ms=activity_times,
        activity_counts=activity_counts,
        top_neurons=top_neurons,
    )
