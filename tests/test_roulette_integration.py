"""Frozen live decisions and bounded trainer/API behavior on a small graph."""

from copy import deepcopy
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import numpy as np
import pytest

from flybrain.connectome import Connectome
from flybrain.roulette_brain import RouletteBrain
from flybrain.roulette_session import RouletteSession
from flybrain.roulette_reports import policy_checksum, save_training_report
from flybrain import roulette_app, roulette_session, roulette_training


def brain_factory(data_dir, seed=42, backend="cpu"):
    n = 64
    sources = np.repeat(np.arange(n), 4)
    graph = Connectome(
        np.arange(n, dtype=np.int64) + 720575940596125868,
        np.arange(0, 4*n+1, 4, dtype=np.int64),
        ((sources + np.tile([1, 3, 7, 13], n)) % n).astype(np.int32),
        (np.where(sources % 3 == 0, -1.0, 1.0) * np.tile([1, 2, 3, 5], n)).astype(np.float32),
    )
    return RouletteBrain(data_dir, seed=seed, graph=graph, backend="cpu")


def ready_session(tmp_path):
    session = RouletteSession(tmp_path, brain_factory=brain_factory)
    session.load_brain("cpu")
    session.worker.join(5)
    assert not session.worker.is_alive()
    assert session.brain is not None and session.error is None
    return session


def test_session_live_fly_actions_freeze_learning_and_use_legal_policy(tmp_path):
    session = ready_session(tmp_path)
    session.new_game(starting_player=1)
    brain = session.brain
    weights, baseline, updates = brain.log_multipliers.copy(), brain.baseline, brain.learning_updates
    steps = 0
    while session.game.winner is None and steps < 50:
        if session.game.turn == 1:
            legal = session.game.legal_actions(1)
            session.fly()
            assert brain.last_action in legal
            assert not brain._pending
        else:
            session.action(session.game.legal_actions(0)[0])
        steps += 1
    assert brain.frames > 0
    np.testing.assert_array_equal(brain.log_multipliers, weights)
    assert brain.baseline == baseline and brain.learning_updates == updates
    assert not session.status()["run"]["busy"]


def test_api_rejects_malformed_mutations_without_changing_game(tmp_path, monkeypatch):
    session = ready_session(tmp_path)
    monkeypatch.setattr(roulette_app, "session", session)
    before = deepcopy(session.status()["game"])
    with TestClient(roulette_app.app) as client:
        for path, body in (
            ("/api/action", {"action": True}), ("/api/action", {"action": 8}),
            ("/api/action", {"action": "0"}), ("/api/action", {"action": 0, "extra": 1}),
            ("/api/train", {"steps": -1}), ("/api/train", {"steps": 20001}),
            ("/api/train", {"steps": 100.0}), ("/api/checkpoint", {"action": "delete"}),
            ("/api/new", {"starting_player": 2}), ("/api/load", {"backend": "cuda"}),
        ):
            response = client.post(path, json=body)
            assert response.status_code == 422, response.text
        response = client.post("/api/fly")
        assert response.status_code == 400
        assert client.get("/api/status").status_code == 200
    assert session.status()["game"] == before
    assert session.training is False and session.loading is False


def test_api_status_never_publishes_the_flys_private_inspection(tmp_path, monkeypatch):
    session = ready_session(tmp_path)
    session.new_game(starting_player=1)
    session.game.inventories[1]["magnifier"] = 1
    session.game.step(2)
    assert session.game.snapshot(1)["known_shell"] in ("live", "blank")
    monkeypatch.setattr(roulette_app, "session", session)
    with TestClient(roulette_app.app) as client:
        view = client.get("/api/status").json()["game"]
    assert view["viewer"] == 0 and view["known_shell"] is None
    event = view["log"][-1]
    assert event["action"] == "magnifier" and "shell" not in event
    assert "_shells" not in view and "shells" not in view


def test_checkpoint_save_load_restores_policy_and_clears_episode(tmp_path):
    session = ready_session(tmp_path)
    brain = session.brain
    session.checkpoint("save")
    saved = brain.log_multipliers.copy()
    obs = session.game.observation(session.game.turn)
    legal = roulette_training.mask_for(session.game)
    target = np.eye(8)[session.game.legal_actions()[0]]
    brain.teach(obs, target, legal_mask=legal)
    assert np.any(brain.log_multipliers != saved)
    session.checkpoint("load")
    np.testing.assert_array_equal(brain.log_multipliers, saved)
    np.testing.assert_array_equal(brain.state, np.full(brain.neuron_count, .5))
    assert not brain._pending


def test_training_excludes_concurrent_mutations_and_stop_reaches_worker(tmp_path, monkeypatch):
    session = ready_session(tmp_path)
    entered, released = threading.Event(), threading.Event()
    def fake_train(brain, output_dir, *, steps, stopped, progress):
        output_dir.mkdir(parents=True)
        progress({"state": "training", "steps": 1})
        entered.set()
        assert released.wait(5)
        assert stopped()
        return {"state": "stopped", "steps": 1}

    monkeypatch.setattr(roulette_session, "train", fake_train)
    session.start_training(100)
    try:
        assert entered.wait(3)
        assert session.status()["run"]["training"] is True
        for operation in (session.new_game, session.fly, lambda: session.action(0),
                          lambda: session.checkpoint("save"), lambda: session.load_brain("cpu"),
                          lambda: session.start_training(100)):
            with pytest.raises(ValueError, match="операции"):
                operation()
        session.stop()
        assert session.stop_event.is_set()
    finally:
        released.set()
        session.worker.join(5)
    assert not session.worker.is_alive()
    assert session.training is False and session.error is None
    assert session.checkpoint_path.is_file()


def test_evaluation_preserves_weights_baseline_rng_and_learning_count():
    brain = brain_factory(Path("."))
    dataset = roulette_training.samples(311, count=12)
    observation, legal, target = dataset[0]
    brain.teach(observation, np.eye(8)[target], legal_mask=legal)
    before = (brain.log_multipliers.copy(), brain.baseline, brain.learning_updates,
              deepcopy(brain.rng.bit_generator.state))
    scores = roulette_training.evaluate(brain, seed=930, games=4)
    for opponent in ("random", "heuristic"):
        result = scores[opponent]
        assert result["wins"] + result["losses"] + result["draws"] == 4
        assert 0 <= result["win_rate"] <= 1
        assert result["fly_decisions"] > 0
    np.testing.assert_array_equal(brain.log_multipliers, before[0])
    assert (brain.baseline, brain.learning_updates) == before[1:3]
    assert brain.rng.bit_generator.state == before[3]
    assert not brain._pending


def test_training_uses_separate_holdout_samples_and_acting_player_masks(tmp_path, monkeypatch):
    brain = brain_factory(tmp_path)
    dataset_ids = {}
    demonstrated = []
    original_samples = roulette_training.samples
    original_teach = brain.teach
    original_mask = roulette_training.mask_for

    def small_samples(seed, count, items):
        result = original_samples(seed, min(count, 32), items)
        dataset_ids[seed] = {id(sample[0]) for sample in result}
        return result

    def record_teach(obs, target_distribution, **kwargs):
        demonstrated.append(id(obs))
        return original_teach(obs, target_distribution, **kwargs)

    def actor_mask(game, player=None):
        assert player is None or player == game.turn
        return original_mask(game, player)

    monkeypatch.setattr(roulette_training, "samples", small_samples)
    monkeypatch.setattr(roulette_training, "mask_for", actor_mask)
    monkeypatch.setattr(brain, "teach", record_teach)
    report = roulette_training.train(brain, tmp_path / "training", steps=40, seed=771,
                                     max_seconds=30, evaluation_games=2)
    assert report["steps"] == 40 and report["episodes"] > 0
    assert report["learning_updates"] == 40
    assert report["weight_change_norm"] > 0
    assert dataset_ids[100771].isdisjoint(demonstrated)
    assert set(demonstrated) <= dataset_ids[771] | dataset_ids[772]
    assert report["heldout_before"]["samples"] == report["heldout_after"]["samples"] == 32
    assert Path(report["checkpoint"]).is_file() and Path(report["report"]).is_file()
    restored = brain_factory(tmp_path)
    restored.load(Path(report["checkpoint"]))
    np.testing.assert_array_equal(restored.log_multipliers, brain.log_multipliers)
    assert report["policy_checksum"] == policy_checksum(restored)


def training_report(brain, **changes):
    return {"schema": "flybrain-roulette-training-v1",
            "graph": {"neurons": brain.neuron_count, "edges": brain.edge_count,
                      "checksum": brain.graph_checksum, "plastic_edges": len(brain.plastic_indices)},
            "cumulative_learning_updates": brain.learning_updates,
            "policy_checksum": policy_checksum(brain),
            "steps": 20, "target_steps": 100, "episodes": 1, "stop_reason": "user_stop", **changes}


@pytest.mark.parametrize("report_text", [None, "{partial", "[]", "null", "{}", '{"schema":"unknown"}'])
def test_optional_report_cannot_block_wake_or_checkpoint_restore(tmp_path, report_text):
    brain = brain_factory(tmp_path)
    output = tmp_path / "outputs"
    checkpoint = output / "roulette-latest.npz"
    brain.save(checkpoint)
    if report_text is not None:
        checkpoint.with_suffix(".json").write_text(report_text)
    session = ready_session(tmp_path)
    assert session.status()["brain"] is not None
    assert session.status()["training"]["state"] == "idle"
    session.new_game(starting_player=1)
    session.fly()
    restored = session.checkpoint("load")
    assert restored["brain"] is not None and restored["run"]["error"] is None
    assert restored["training"]["state"] == "idle"


@pytest.mark.parametrize("change", [
    {"cumulative_learning_updates": 99}, {"graph": {}},
    {"policy_checksum": "stale-weights"}, {"steps": "20"},
    {"teacher_agreement_after": float("nan")}, {"evaluation": {"score": float("inf")}},
])
def test_stale_training_evidence_is_not_attached_to_loaded_weights(tmp_path, change):
    brain = brain_factory(tmp_path)
    checkpoint = tmp_path / "outputs" / "roulette-latest.npz"
    brain.save(checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps(training_report(brain, **change)))
    session = ready_session(tmp_path)
    assert session.status()["training"]["state"] == "idle"
    assert session.status()["run"]["error"] is None


@pytest.mark.parametrize("legacy", [False, True])
def test_matching_report_survives_live_actions_and_manual_save(tmp_path, legacy):
    session = ready_session(tmp_path)
    report = training_report(session.brain)
    if legacy:
        del report["policy_checksum"]
    session.checkpoint("save")
    save_training_report(session.checkpoint_path.with_suffix(".json"), report)
    session.new_game(starting_player=1)
    session.fly()
    session.checkpoint("save")
    restored = ready_session(tmp_path)
    assert restored.status()["training"] == {**report, "state": "stopped"}
    assert restored.checkpoint("load")["training"] == {**report, "state": "stopped"}


def test_changed_policy_with_same_update_count_rejects_new_report(tmp_path):
    brain = brain_factory(tmp_path)
    report = training_report(brain)
    brain.log_multipliers[0] += .01
    checkpoint = tmp_path / "outputs" / "roulette-latest.npz"
    brain.save(checkpoint)
    save_training_report(checkpoint.with_suffix(".json"), report)
    assert ready_session(tmp_path).training_status["state"] == "idle"


def test_failed_checkpoint_wake_can_retry_after_file_is_repaired(tmp_path):
    checkpoint = tmp_path / "outputs" / "roulette-latest.npz"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"incomplete checkpoint")
    session = RouletteSession(tmp_path, brain_factory=brain_factory)
    session.load_brain("cpu")
    session.worker.join(5)
    assert session.brain is None and session.brain_status is None
    assert session.error is not None and not session.loading
    brain_factory(tmp_path).save(checkpoint)
    session.load_brain("cpu")
    session.worker.join(5)
    assert session.status()["brain"] is not None and session.error is None


def test_report_write_failure_preserves_previous_json_and_cleans_temporary_file(tmp_path, monkeypatch):
    from flybrain import roulette_reports

    path = tmp_path / "report.json"
    save_training_report(path, {"previous": True})

    def failed_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(roulette_reports.os, "replace", failed_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        save_training_report(path, {"new": True})
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]
