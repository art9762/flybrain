from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from flybrain.brain import Brain
from flybrain.connectome import Connectome


def graph_fixture(n=32) -> Connectome:
    # Ring plus longer links: all 32 real fixture nodes have incoming edges,
    # with inhibitory and excitatory sources and no invented trainable edges.
    sources = np.repeat(np.arange(n), 4)
    targets = (sources + np.tile([1, 3, 7, 13], n)) % n
    signs = np.where(sources % 3 == 0, -1.0, 1.0)
    return Connectome(
        np.arange(n, dtype=np.int64) + 720575940596125868,
        np.arange(0, 4 * n + 1, 4, dtype=np.int64),
        targets.astype(np.int32),
        (signs * np.tile([1, 2, 3, 5], n)).astype(np.float32),
    )


def make_brain() -> Brain:
    return Brain(Path("."), seed=12, graph=graph_fixture(), backend="cpu")


def image() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)


def test_existing_signed_edges_change_and_stay_bounded():
    brain = make_brain()
    original = brain.matrix.data.copy()
    original_indices, original_indptr = brain.matrix.indices.copy(), brain.matrix.indptr.copy()
    for step in range(50):
        brain.act(image())
        brain.feedback(20.0 if step < 25 else -20.0)
    assert np.count_nonzero(brain.matrix.data != original) > 0
    np.testing.assert_array_equal(np.sign(brain.matrix.data), np.sign(original))
    np.testing.assert_array_equal(brain.matrix.indices, original_indices)
    np.testing.assert_array_equal(brain.matrix.indptr, original_indptr)
    ratio = brain.matrix.data / original
    assert ratio.min() >= 0.1 - 1e-6
    assert ratio.max() <= 3.0 + 1e-6
    fixed = np.ones(brain.edge_count, dtype=bool)
    fixed[brain.plastic_indices] = False
    np.testing.assert_array_equal(brain.matrix.data[fixed], original[fixed])
    assert brain.observe_status()["activeNeurons"] == brain.neuron_count


def test_disabled_learning_freezes_weights_critic_and_traces():
    brain = make_brain()
    brain.act(image())
    brain.feedback(1.0)
    before = (brain.matrix.data.copy(), brain.eligibility.copy(), brain.baseline)
    brain.act(image(), learn=False)
    brain.feedback(10.0)
    np.testing.assert_array_equal(brain.matrix.data, before[0])
    np.testing.assert_array_equal(brain.eligibility, before[1])
    assert brain.baseline == before[2]
    brain.act(image(), learn=True)
    brain.feedback(10.0, learn=False)
    np.testing.assert_array_equal(brain.matrix.data, before[0])


def test_busy_ticks_preserve_initiating_trace_and_still_run_all_recurrent_steps():
    brain = make_brain()
    brain.baseline = 0.25
    brain.act(image())
    brain.feedback(0, action_effective=True, outcome_ready=False)
    initial = (brain.eligibility.copy(), brain.matrix.data.copy(), brain.baseline,
               brain.frames, brain.neural_steps, brain.learning_updates)
    assert np.any(initial[0])
    assert brain.observe_status()["canReinforce"] is True

    for _ in range(10):
        brain.act(image())
        brain.feedback(0, action_effective=False, outcome_ready=False)
        np.testing.assert_array_equal(brain.eligibility, initial[0])
        np.testing.assert_array_equal(brain.matrix.data, initial[1])
        assert brain.baseline == initial[2]
        assert brain.observe_status()["canReinforce"] is True

    assert brain.frames == initial[3] + 10
    assert brain.neural_steps == initial[4] + 10 * brain.RECURRENT_STEPS
    assert brain.learning_updates == initial[5]
    brain.reinforce(0.1)
    assert np.any(brain.matrix.data != initial[1])
    np.testing.assert_array_equal(brain.eligibility, initial[0])


@pytest.mark.parametrize("reward", (-0.1, 0.0, 0.1))
def test_delayed_outcome_updates_retained_trace_like_immediate_outcome(tmp_path, reward):
    immediate = make_brain()
    immediate.baseline = 0.03
    immediate.act(image())
    immediate.feedback(0, action_effective=True, outcome_ready=False)
    initial_trace = immediate.eligibility.copy()
    initial_weights = immediate.matrix.data.copy()
    checkpoint = tmp_path / "started-action.npz"
    immediate.save(checkpoint)
    delayed = make_brain()
    delayed.load(checkpoint)

    immediate.act(image())
    immediate.feedback(reward, action_effective=False, outcome_ready=True)
    for _ in range(10):
        delayed.act(image())
        delayed.feedback(0, action_effective=False, outcome_ready=False)
    delayed.act(image())
    delayed.feedback(reward, action_effective=False, outcome_ready=True)

    np.testing.assert_array_equal(delayed.eligibility, initial_trace)
    np.testing.assert_array_equal(delayed.matrix.data, immediate.matrix.data)
    assert np.any(delayed.matrix.data != initial_weights)  # Zero reward has a nonzero baseline.
    assert delayed.baseline == immediate.baseline
    assert delayed.learning_updates == immediate.learning_updates == 1
    assert delayed.total_reward == immediate.total_reward == reward


def test_effective_choice_during_script_updates_trace_without_premature_outcome():
    brain = make_brain()
    brain.act(image())
    brain.feedback(0, action_effective=True, outcome_ready=False)
    trace = brain.eligibility.copy()
    brain.act(image())
    gradient = brain._pending_gradient.copy()
    brain.feedback(0, action_effective=True, outcome_ready=False)
    np.testing.assert_allclose(brain.eligibility, trace * brain.TRACE_DECAY + gradient)
    assert brain.learning_updates == 0
    assert brain.baseline == 0


def test_feedback_flags_keep_reward_accounting_and_evaluation_frozen():
    brain = make_brain()
    brain.act(image())
    brain.feedback(0, action_effective=True, outcome_ready=False)
    initial = (brain.eligibility.copy(), brain.matrix.data.copy(), brain.baseline)
    brain.act(image(), learn=False)
    brain.feedback(0.4, action_effective=False, outcome_ready=True)
    np.testing.assert_array_equal(brain.eligibility, initial[0])
    np.testing.assert_array_equal(brain.matrix.data, initial[1])
    assert brain.baseline == initial[2]
    assert brain.last_reward == brain.total_reward == 0.4
    assert brain.learning_updates == 0
    assert brain.observe_status()["canReinforce"] is False


def test_reenabled_learning_does_not_credit_an_ignored_completion_to_stale_trace():
    brain = make_brain()
    brain.act(image())
    brain.feedback(0, action_effective=True, outcome_ready=False)
    trace = brain.eligibility.copy()
    weights = brain.matrix.data.copy()
    brain.act(image(), learn=False)
    brain.feedback(0, action_effective=False, outcome_ready=False)
    brain.act(image(), learn=True)
    brain.feedback(1, action_effective=False, outcome_ready=True)
    np.testing.assert_array_equal(brain.eligibility, trace)
    np.testing.assert_array_equal(brain.matrix.data, weights)
    assert brain.total_reward == 1
    assert brain.baseline == 0
    assert brain.learning_updates == 0
    with pytest.raises(RuntimeError, match="completed training"):
        brain.reinforce(1)


def test_accounting_does_not_drop_reward_before_deferred_outcome():
    brain = make_brain()
    brain.act(image())
    brain.feedback(0.2, action_effective=True, outcome_ready=False)
    assert brain.last_reward == brain.total_reward == 0.2
    assert brain.baseline == 0
    assert brain.learning_updates == 0


def test_rgb_changes_neural_activity_and_has_strict_observation_contract():
    a = make_brain()
    b = make_brain()
    red, blue = np.zeros((48, 64, 3), dtype=np.uint8), np.zeros((48, 64, 3), dtype=np.uint8)
    red[..., 0], blue[..., 2] = 255, 255
    a.act(red, learn=False)
    b.act(blue, learn=False)
    assert not np.allclose(a.state, b.state)
    assert not np.allclose(a.action_probabilities, b.action_probabilities)
    assert not np.allclose(Brain.encode_vision(red), Brain.encode_vision(blue))
    with pytest.raises(ValueError, match="shape"):
        Brain.encode_vision(np.zeros((48, 64, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="finite"):
        Brain.encode_vision(np.full((48, 64, 3), np.nan))


def test_checkpoint_replays_pending_update_actions_and_rng(tmp_path):
    a = make_brain()
    a.act(image())
    path = tmp_path / "brain.npz"
    a.save(path)
    b = make_brain()
    b.load(path)
    for reward in (0.5, -2.0, 1.0):
        a.feedback(reward)
        b.feedback(reward)
        np.testing.assert_array_equal(a.log_multipliers, b.log_multipliers)
        np.testing.assert_array_equal(a.eligibility, b.eligibility)
        assert a.act(image()) == b.act(image())
        np.testing.assert_array_equal(a.state, b.state)
        np.testing.assert_array_equal(a.action_probabilities, b.action_probabilities)
    assert a.observe_status() == b.observe_status()
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)


def test_checkpoint_rejects_wrong_source_and_corruption_before_mutation(tmp_path):
    a = make_brain()
    a.act(image())
    path = tmp_path / "brain.npz"
    a.save(path)
    wrong_graph = graph_fixture()
    wrong_graph.weights[0] *= 2
    other = Brain(Path("."), seed=12, graph=wrong_graph, backend="cpu")
    with pytest.raises(ValueError, match="graph"):
        other.load(path)
    np.testing.assert_array_equal(other.state, np.full(other.neuron_count, 0.5))
    with np.load(path, allow_pickle=False) as archive:
        content = {key: archive[key] for key in archive.files}
    content["state"][0] += 0.01
    np.savez(tmp_path / "corrupt.npz", **content)
    before = a.state.copy()
    with pytest.raises(ValueError, match="checksum"):
        a.load(tmp_path / "corrupt.npz")
    np.testing.assert_array_equal(a.state, before)


def test_local_policy_derivative_matches_finite_difference_and_reward_increases_it():
    brain = make_brain()
    previous = np.linspace(0.05, 0.9, brain.neuron_count, dtype=np.float32)
    drive = np.zeros(brain.neuron_count, dtype=np.float32)
    rates = expit(brain.matrix @ (2 * previous - 1) + drive + brain.BIAS)
    state = (1 - brain.LEAK) * previous + brain.LEAK * rates
    probabilities = brain._probabilities(state)
    action = 2
    gradient = brain._local_score(action, previous, rates, probabilities)
    # Derivative of the final neural step only; earlier state is held fixed.
    edge = int(np.argmax(np.abs(gradient)))
    location = brain.plastic_indices[edge]
    weight = float(brain.matrix.data[location])

    def log_probability(offset: float) -> float:
        brain.matrix.data[location] = weight * np.exp(offset)
        current_rates = expit(brain.matrix @ (2 * previous - 1) + drive + brain.BIAS)
        current_state = (1 - brain.LEAK) * previous + brain.LEAK * current_rates
        return float(np.log(brain._probabilities(current_state)[action]))

    difference = (log_probability(0.01) - log_probability(-0.01)) / 0.02
    brain.matrix.data[location] = weight
    assert gradient[edge] == pytest.approx(difference, rel=0.01, abs=1e-5)
    before = float(np.log(probabilities[action]))
    brain._pending_gradient[:] = gradient
    brain._pending = brain._pending_learn = True
    brain.feedback(0.1)
    after_rates = expit(brain.matrix @ (2 * previous - 1) + drive + brain.BIAS)
    after_state = (1 - brain.LEAK) * previous + brain.LEAK * after_rates
    assert np.log(brain._probabilities(after_state)[action]) > before


def test_action_feedback_order_and_invalid_reward():
    brain = make_brain()
    with pytest.raises(RuntimeError, match="preceding"):
        brain.feedback(1)
    brain.act(image())
    with pytest.raises(RuntimeError, match="feedback"):
        brain.act(image())
    with pytest.raises(ValueError, match="finite"):
        brain.feedback(float("nan"))
    brain.feedback(0)


def test_manual_reward_uses_recent_trace_without_an_action_and_survives_save(tmp_path):
    brain = make_brain()
    with pytest.raises(RuntimeError, match="completed training"):
        brain.reinforce(1)
    brain.act(image())
    brain.feedback(0)
    old_weights = brain.matrix.data.copy()
    old_trace = brain.eligibility.copy()
    old_frames, old_steps = brain.frames, brain.neural_steps
    brain.reinforce(0.5)
    assert np.count_nonzero(old_weights != brain.matrix.data) > 0
    np.testing.assert_array_equal(np.sign(old_weights), np.sign(brain.matrix.data))
    np.testing.assert_array_equal(old_trace, brain.eligibility)
    assert (old_frames, old_steps) == (brain.frames, brain.neural_steps)
    assert brain.baseline == pytest.approx(brain.BASELINE_RATE * 0.5)
    assert brain.observe_status()["manualRewardTotal"] == 0.5
    checkpoint = tmp_path / "manual.npz"
    brain.save(checkpoint)
    restored = make_brain()
    restored.load(checkpoint)
    brain.reinforce(-0.25)
    restored.reinforce(-0.25)
    assert brain.observe_status() == restored.observe_status()
    np.testing.assert_array_equal(brain.matrix.data, restored.matrix.data)
    brain.act(image(), learn=False)
    brain.feedback(1)
    with pytest.raises(RuntimeError, match="completed training"):
        brain.reinforce(1)
    restored.reset_activity()
    with pytest.raises(RuntimeError, match="completed training"):
        restored.reinforce(1)


def test_black_image_preserves_neutral_activity_without_hidden_inputs():
    brain = make_brain()
    brain.act(np.zeros((48, 64, 3), dtype=np.uint8), learn=False)
    np.testing.assert_array_equal(brain.state, np.full(brain.neuron_count, 0.5))
    np.testing.assert_array_equal(brain.action_probabilities, np.full(8, 0.125))


def test_reward_learns_image_specific_association_in_fixture():
    # A synthetic controller regression; full-FlyWire acceptance is a separate
    # bounded runtime probe. Labels live exclusively in this reward evaluator.
    brain = Brain(Path("."), seed=42, graph=graph_fixture(64), backend="cpu")
    pictures = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    pictures[0, 8:40, 4:28, 0] = 255
    pictures[1, 8:40, 36:60, 2] = 255

    def score():
        correct, swapped = [], []
        noise_rng = np.random.default_rng(8091)
        for label in range(2):
            pixels = np.clip(pictures[label].astype(float) + noise_rng.normal(0, 8, pictures[label].shape), 0, 255).astype(np.uint8)
            brain.reset_activity()
            brain.act(pixels, learn=False)
            correct.append(brain.action_probabilities[label + 1])
            swapped.append(brain.action_probabilities[2 - label])
            brain.feedback(0, learn=False)
        return float(np.mean(correct)), float(np.mean(correct) - np.mean(swapped))

    before_reward, before_margin = score()
    curriculum = np.random.default_rng(59)
    for _ in range(800):
        label = int(curriculum.integers(0, 2))
        brain.reset_activity()
        action = brain.act(pictures[label])
        brain.feedback(float(action == label + 1))
    after_reward, after_margin = score()
    assert after_reward > before_reward + 0.15
    assert after_margin > before_margin + 0.15
    assert after_margin > 0.25
