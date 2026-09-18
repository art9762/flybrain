"""Controller checks on an injected fixture, not full-FlyWire acceptance."""

from pathlib import Path

import numpy as np
import pytest

from flybrain.brain import Brain
from flybrain.connectome import Connectome
from flybrain.roulette_brain import ACTIONS, RouletteBrain


def graph_fixture(n=64):
    sources = np.repeat(np.arange(n), 4)
    return Connectome(
        np.arange(n, dtype=np.int64) + 720575940596125868,
        np.arange(0, 4 * n + 1, 4, dtype=np.int64),
        ((sources + np.tile([1, 3, 7, 13], n)) % n).astype(np.int32),
        (np.where(sources % 3 == 0, -1.0, 1.0) * np.tile([1, 2, 3, 5], n)).astype(np.float32),
    )


def make_brain():
    return RouletteBrain(Path("."), seed=42, graph=graph_fixture(), backend="cpu")


def observation():
    return np.random.default_rng(231).uniform(-1, 1, 48).astype(np.float32)


def test_numeric_input_changes_entire_fixture_state_and_action_probabilities():
    a, b = make_brain(), make_brain()
    a.act(observation())
    b.act(-observation())
    assert not np.allclose(a.state, b.state)
    assert not np.allclose(a.action_probabilities, b.action_probabilities)
    assert a.neuron_count == 64 and a.edge_count == 256
    assert a.neural_steps == a.RECURRENT_STEPS
    assert a.observe_status()["actions"] == list(ACTIONS)
    assert a.encode_vision(observation()).shape == (768,)


@pytest.mark.parametrize("bad", [np.zeros((48, 64, 3), dtype=np.uint8), np.zeros(48, dtype=int),
                                      np.full(48, np.nan), np.full(48, 1.1)])
def test_observations_are_strict_and_reject_before_neural_mutation(bad):
    brain = make_brain()
    initial = brain.state.copy()
    with pytest.raises(ValueError, match="observation"):
        brain.act(bad)
    np.testing.assert_array_equal(brain.state, initial)
    assert brain.frames == 0 and not brain._pending


def test_legal_mask_prevents_illegal_actions_and_handles_single_option():
    brain = make_brain()
    mask = np.asarray([False, False, False, True, False, False, False, False])
    for _ in range(5):
        assert brain.act(observation(), legal_mask=mask) == 3
        np.testing.assert_array_equal(brain.action_probabilities, mask.astype(float))
        brain.feedback(0, learn=False)
    for bad in (np.zeros(8, dtype=bool), np.ones(8, dtype=int), np.ones(7, dtype=bool)):
        with pytest.raises(ValueError, match="legal_mask"):
            brain.act(observation(), legal_mask=bad)


def test_supervision_improves_association_preserving_graph_and_signs():
    brain = make_brain()
    original = brain.matrix.data.copy()
    indices, pointers = brain.matrix.indices.copy(), brain.matrix.indptr.copy()
    pictures = np.stack([observation(), -observation()])
    mask = np.asarray([True, True, False, False, False, False, False, False])

    def score():
        correct = []
        for label, obs in enumerate(pictures):
            brain.reset_activity()
            brain.act(obs, legal_mask=mask)
            correct.append(brain.action_probabilities[label])
            brain.feedback(0, learn=False)
        return float(np.mean(correct))

    before = score()
    rng_state = brain.rng.bit_generator.state
    for i in range(400):
        label = i % 2
        target = np.eye(8)[label]
        metrics = brain.teach(pictures[label], target, legal_mask=mask)
        assert metrics["loss"] >= 0 and 0 <= metrics["targetprob"] <= 1
        assert not brain._pending
    assert brain.rng.bit_generator.state == rng_state
    assert score() > before + 0.15
    assert np.any(brain.matrix.data != original)
    np.testing.assert_array_equal(brain.matrix.indices, indices)
    np.testing.assert_array_equal(brain.matrix.indptr, pointers)
    np.testing.assert_array_equal(np.sign(brain.matrix.data), np.sign(original))
    fixed = np.ones(brain.edge_count, dtype=bool)
    fixed[brain.plastic_indices] = False
    np.testing.assert_array_equal(brain.matrix.data[fixed], original[fixed])
    assert brain.baseline == 0 and brain.total_reward == 0
    assert brain.learning_updates == 400


def test_default_evaluation_freezes_all_learning_state():
    brain = make_brain()
    brain.teach(observation(), np.eye(8)[0])
    original = (brain.matrix.data.copy(), brain.eligibility.copy(), brain.baseline, brain.learning_updates)
    brain.act(-observation())
    brain.feedback(100)
    np.testing.assert_array_equal(brain.matrix.data, original[0])
    np.testing.assert_array_equal(brain.eligibility, original[1])
    assert brain.baseline == original[2]
    assert brain.learning_updates == original[3]
    with pytest.raises(RuntimeError, match="completed training"):
        brain.reinforce(1)


def test_reward_learning_remains_available_and_teaching_requires_completed_feedback():
    brain = make_brain()
    brain.act(observation(), learn=True)
    with pytest.raises(RuntimeError, match="feedback"):
        brain.teach(observation(), np.eye(8)[0])
    brain.feedback(0.5)
    assert brain.learning_updates == 1
    brain.teach(observation(), np.eye(8)[0])
    assert brain.learning_updates == 2
    np.testing.assert_array_equal(brain.eligibility, np.zeros_like(brain.eligibility))


def test_invalid_teacher_targets_reject_before_reset():
    brain = make_brain()
    brain.act(observation())
    brain.feedback(0, learn=False)
    state = brain.state.copy()
    mask = np.asarray([True, True, False, False, False, False, False, False])
    for bad in (np.zeros(8), np.ones(7) / 7, np.full(8, np.nan), np.eye(8)[3]):
        with pytest.raises(ValueError, match="target_distribution"):
            brain.teach(observation(), bad, legal_mask=mask)
        np.testing.assert_array_equal(brain.state, state)


def test_checkpoint_replays_pending_feedback_and_rejects_room_brain(tmp_path):
    a = make_brain()
    a.teach(observation(), np.eye(8)[0])
    mask = np.asarray([True, True, False, False, False, False, False, False])
    a.act(observation(), learn=True, legal_mask=mask)
    path = tmp_path / "roulette.npz"
    a.save(path)
    b = make_brain()
    b.load(path)
    for reward in (0.3, -0.4):
        a.feedback(reward)
        b.feedback(reward)
        np.testing.assert_array_equal(a.matrix.data, b.matrix.data)
        assert a.act(-observation(), learn=True, legal_mask=mask) == b.act(-observation(), learn=True, legal_mask=mask)
        np.testing.assert_array_equal(a.action_probabilities, b.action_probabilities)
    room = Brain(Path("."), seed=42, graph=graph_fixture(), backend="cpu")
    with pytest.raises(ValueError, match="model"):
        room.load(path)
    room.save(tmp_path / "room.npz")
    with pytest.raises(ValueError, match="model"):
        make_brain().load(tmp_path / "room.npz")
