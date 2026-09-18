import numpy as np

from flybrain.connectome import Connectome, extract_reachable_subgraph
from flybrain.simulator import SimulationConfig, simulate


def tiny_graph() -> Connectome:
    return Connectome(
        neuron_ids=np.array([101, 102, 103], dtype=np.int64),
        indptr=np.array([0, 1, 2, 2], dtype=np.int64),
        targets=np.array([1, 2], dtype=np.int32),
        weights=np.array([20.0, 20.0], dtype=np.float32),
    )


def test_extract_subgraph_keeps_reachable_neurons_and_ids():
    subgraph = extract_reachable_subgraph(tiny_graph(), seed=0, maximum_neurons=2, maximum_hops=3)
    assert subgraph.neuron_ids.tolist() == [101, 102]
    assert subgraph.targets.tolist() == [1]


def test_stimulus_produces_seed_spikes():
    result = simulate(
        tiny_graph(),
        seed_index=0,
        config=SimulationConfig(duration_ms=20, stimulus_start_ms=1, stimulus_end_ms=10),
    )
    assert result.spike_count > 0
    assert 101 in result.spike_flywire_ids


def test_invalid_stimulus_interval_is_rejected():
    config = SimulationConfig(duration_ms=10, stimulus_start_ms=5, stimulus_end_ms=20)
    try:
        simulate(tiny_graph(), 0, config)
    except ValueError as error:
        assert "stimulus interval" in str(error)
    else:
        raise AssertionError("invalid config was accepted")
