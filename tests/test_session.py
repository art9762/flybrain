"""Small-model session integration checks; never load the production graph."""

import base64
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from flybrain import brain as brain_module
from flybrain import lab
from flybrain.session import LabSession
from flybrain.world import ACTIONS


class RecordingBrain:
    def __init__(self, data_dir, seed, backend):
        self.backend = backend
        self.observations = []
        self.feedback_calls = []
        self.feedback_flags = []
        self.reinforcements = []
        self.pending = False
        self.can_reinforce = False
        self.act_entered = threading.Event()
        self.act_release = None

    def act(self, image, learn=True):
        assert isinstance(image, np.ndarray)
        assert image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
        assert not self.pending
        self.act_entered.set()
        if self.act_release is not None:
            assert self.act_release.wait(2), "test did not release blocked action"
        self.observations.append((image.copy(), learn))
        self.pending = True
        return ACTIONS.index("forward")

    def feedback(self, reward, learn=True, *, action_effective=True, outcome_ready=True):
        assert self.pending
        self.feedback_calls.append((reward, learn))
        self.feedback_flags.append((action_effective, outcome_ready))
        self.pending = False
        self.can_reinforce = bool(learn)

    def reset_activity(self):
        self.can_reinforce = False
        self.pending = False

    def reinforce(self, amount):
        if not self.can_reinforce or self.pending:
            raise RuntimeError("manual reward requires a completed training action")
        assert np.isfinite(amount)
        self.reinforcements.append(amount)

    def observe_status(self):
        return {"canReinforce": self.can_reinforce, "frames": len(self.observations),
                "backend": {"selected": self.backend}}


def until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for session state")
        threading.Event().wait(0.005)


@pytest.fixture
def make_session(monkeypatch, tmp_path):
    sessions = []
    releases = []

    def make(factory=RecordingBrain):
        monkeypatch.setattr(brain_module, "Brain", factory)
        session = LabSession(tmp_path)
        sessions.append(session)
        return session

    make.releases = releases
    yield make
    for release in releases:
        release.set()
    for session in sessions:
        # There is no public shutdown API yet. Wake the idle daemon with a
        # test-only wait exception, so each test leaves no background worker.
        with session.condition:
            session.running = False
            session.pending_steps = 0

            def stopped(timeout=None):
                raise RuntimeError("test session closed")

            session.condition.wait = stopped
            session.condition.notify_all()
        if session.worker is not None:
            session.worker.join(timeout=2)
            assert not session.worker.is_alive(), "session worker outlived its test"


def test_one_step_routes_only_pixels_and_pairs_feedback(make_session, monkeypatch):
    session = make_session()
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    monkeypatch.setattr(session.world, "frame", lambda: pixels.copy())
    start_z = session.world.z
    session.control("step", learning=True)
    until(lambda: session.tick_count == 1)
    paused = session.control("pause")
    brain = session.brain
    assert len(brain.observations) == 1
    assert np.array_equal(brain.observations[0][0], pixels)
    assert brain.observations[0][1] is True
    assert brain.feedback_calls == [(0.0, True)]
    assert session.world.z == pytest.approx(start_z - 0.32)
    assert paused["run"]["last_action"] == "forward"
    assert base64.b64decode(paused["eye"]["rgb"]) == pixels.tobytes()


def test_manual_reward_targets_completed_transition_without_another_step(make_session):
    session = make_session()
    with pytest.raises(ValueError):
        session.reward(1)
    session.control("step", learning=True)
    until(lambda: session.tick_count == 1)
    session.control("pause")
    session.reward(2)
    result = session.reward(-0.5)
    assert session.tick_count == 1
    assert len(session.brain.observations) == 1
    assert session.brain.reinforcements == [2, -0.5]
    assert result["run"]["manual_reward"] == 1.5
    assert [entry["source"] for entry in result["manual_events"]] == ["human", "human"]
    assert result["world"]["total_reward"] == 0  # Human and automatic rewards are separate.
    session.control("pause", learning=False)
    with pytest.raises(ValueError):
        session.reward(1)


def test_loading_is_nonblocking_and_pause_cancels_queued_steps(make_session):
    entered, release = threading.Event(), threading.Event()
    make_session.releases.append(release)

    def slow_brain(**kwargs):
        entered.set()
        assert release.wait(2), "test did not release model loader"
        return RecordingBrain(**kwargs)

    session = make_session(slow_brain)
    started = time.monotonic()
    initial = session.control("step")
    assert time.monotonic() - started < 0.5
    assert initial["run"]["loading"] is True
    assert entered.wait(1)
    status_started = time.monotonic()
    status = session.status()
    assert time.monotonic() - status_started < 0.5
    assert status["run"]["loading"] is True
    assert status["brain"] is None
    session.control("step")
    session.control("step")
    with pytest.raises(ValueError):
        session.control("pause", backend="cpu")
    session.control("pause")
    release.set()
    until(lambda: session.brain is not None)
    assert session.tick_count == 0
    assert session.pending_steps == 0
    session.control("step")
    until(lambda: session.tick_count == 1)
    assert session.control("pause")["run"]["steps"] == 1
    with pytest.raises(ValueError):
        session.control("pause", backend="mps")


def test_pause_serializes_with_inflight_action_and_stops_future_steps(make_session):
    release = threading.Event()
    make_session.releases.append(release)

    def gated_brain(**kwargs):
        brain = RecordingBrain(**kwargs)
        brain.act_release = release
        return brain

    session = make_session(gated_brain)
    session.control("start", speed=1)
    until(lambda: session.brain is not None)
    assert session.brain.act_entered.wait(1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        paused_future = pool.submit(session.control, "pause")
        assert not paused_future.done()  # An in-flight transaction holds the lock.
        release.set()
        paused = paused_future.result(timeout=1)
    assert paused["run"]["running"] is False
    assert paused["run"]["steps"] == 1
    threading.Event().wait(0.3)  # More than one normal 0.25-second interval.
    assert session.status()["run"]["steps"] == 1
    assert len(session.brain.feedback_calls) == 1


def test_http_rejects_nonfinite_inputs_and_unknown_fields(make_session, monkeypatch):
    session = make_session()
    monkeypatch.setattr(lab, "session", session)
    with TestClient(lab.app) as client:
        for path, field in (("/api/reward", "amount"), ("/api/control", "speed")):
            for value in ("NaN", "Infinity", "-Infinity"):
                prefix = '"action":"pause",' if field == "speed" else ""
                response = client.post(path, content="{" + prefix + f'"{field}":"{value}"' + "}",
                                       headers={"Content-Type": "application/json"})
                assert response.status_code == 422
        assert client.post("/api/control", json={"action": "launch"}).status_code == 422
        assert client.post("/api/reward", json={"amount": 1, "secret": True}).status_code == 422
        assert client.post("/api/credits", json={"amount": 0}).status_code == 422
        assert client.post("/api/credits", json={"amount": 1.5}).status_code == 422
        assert client.post("/api/reward", json={"amount": 1}).status_code == 400
    assert session.brain is None
    assert session.world.bank == 100


def test_reenabling_learning_after_evaluation_requires_new_training_transition(make_session, monkeypatch):
    session = make_session()
    monkeypatch.setattr(lab, "session", session)
    session.control("step", learning=False)
    until(lambda: session.tick_count == 1)
    session.control("pause", learning=True)
    with TestClient(lab.app, raise_server_exceptions=False) as client:
        response = client.post("/api/reward", json={"amount": 1})
    assert response.status_code == 400
    assert session.brain.reinforcements == []
    assert session.manual_total == 0


def test_overflowing_json_numbers_return_validation_error_not_server_error(make_session, monkeypatch):
    session = make_session()
    monkeypatch.setattr(lab, "session", session)
    with TestClient(lab.app, raise_server_exceptions=False) as client:
        requests = (("/api/reward", '{"amount":1e309}'),
                    ("/api/control", '{"action":"pause","speed":-1e309}'),
                    ("/api/reward", '{"amount":NaN}'))
        for path, body in requests:
            response = client.post(path, content=body, headers={"Content-Type": "application/json"})
            assert response.status_code == 422
    assert session.brain is None
    assert session.manual_total == 0


def test_failed_backend_change_does_not_mutate_other_control_settings(make_session):
    session = make_session()
    session.control("step", learning=True, speed=1)
    until(lambda: session.tick_count == 1)
    session.control("pause")
    with pytest.raises(ValueError):
        session.control("start", learning=False, speed=8, backend="cpu")
    assert session.learning is True
    assert session.speed == 1
    assert session.running is False


def test_reset_position_preserves_brain_rewards_and_bank_and_pauses(make_session, monkeypatch):
    session = make_session()
    session.control("step")
    until(lambda: session.tick_count == 1)
    session.control("pause")
    session.reward(2)
    session.credits(100)
    brain = session.brain
    session.world.x = 3
    before = session.status()
    monkeypatch.setattr(lab, "session", session)
    with TestClient(lab.app) as client:
        response = client.post("/api/reset-position")
    assert response.status_code == 200
    after = response.json()
    assert after["world"]["fly"] == {"x": 0, "z": 1, "heading": 0}
    assert after["world"]["bank"] == before["world"]["bank"] == 200
    assert after["world"]["time_s"] == before["world"]["time_s"]
    assert after["world"]["total_reward"] == before["world"]["total_reward"]
    assert after["run"]["steps"] == before["run"]["steps"]
    assert after["run"]["manual_reward"] == 2
    assert after["run"]["running"] is False
    assert session.brain is brain
    assert brain.reinforcements == [2]
    assert not brain.can_reinforce
    with pytest.raises(ValueError):
        session.reward(1)
    session.control("step")
    until(lambda: session.tick_count == 2)
    session.control("pause")
    assert session.world.z == pytest.approx(0.68)


def test_reset_position_works_before_brain_load_and_rejects_during_loading(make_session):
    session = make_session()
    session.world.x = 4
    assert session.reset_position()["world"]["fly"]["x"] == 0
    assert session.brain is None
    session.loading = True
    session.world.x = 2
    with pytest.raises(ValueError, match="загрузки"):
        session.reset_position()
    assert session.world.x == 2


def test_session_ignores_busy_choices_but_routes_completion_to_retained_choice(make_session):
    class InteractingBrain(RecordingBrain):
        def act(self, image, learn=True):
            super().act(image, learn)
            return ACTIONS.index("interact" if len(self.observations) == 1 else "forward")

    session = make_session(InteractingBrain)
    session.world.x, session.world.z = 3, -2
    for tick in range(10):
        session.control("step")
        until(lambda: session.tick_count == tick + 1)
        session.control("pause")
    assert session.world.stations["hookah"]["uses"] == 1
    assert session.brain.feedback_flags[0] == (True, False)
    assert session.brain.feedback_flags[1:-1] == [(False, False)] * 8
    assert session.brain.feedback_flags[-1] == (False, True)
    assert session.last_effective_action == "interact"
    assert session.last_action == "busy"
    session.reward(1)
    assert session.manual_events[-1]["action"] == "interact"
