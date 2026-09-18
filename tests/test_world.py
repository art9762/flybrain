import json
import math

import numpy as np
import pytest

from flybrain.world import ACTIONS, World


def action(world, name):
    return world.step(ACTIONS.index(name))


def approach(world, sid):
    station = world.stations[sid]
    world.x, world.z, world.heading = station["x"], station["z"] + 1.0, 0.0


def finish(world):
    rewards = []
    while world.pending is not None:
        rewards.append(action(world, "wait")["reward"])
    return sum(rewards)


class FixedOutcome:
    def __init__(self, number):
        self.number = number
        self.calls = 0

    def integers(self, low, high):
        assert (low, high) == (0, 37)
        self.calls += 1
        return self.number


@pytest.mark.parametrize("bet", ["bet_red", "bet_black"])
def test_zero_loses_and_stake_is_deducted_once(bet):
    world = World()
    world.roulette_rng = FixedOutcome(0)
    approach(world, "roulette")
    result = action(world, bet)
    assert result["reward"] == 0
    assert world.bank == 95
    assert world.roulette_rng.calls == 0
    assert "number" not in world.snapshot()["last_interaction"]
    action(world, bet)  # An in-progress gesture cannot place another bet.
    assert world.bank == 95
    assert finish(world) == -5
    assert world.bank == 95
    assert world.roulette_rng.calls == 1
    assert world.last_interaction["number"] == 0
    assert world.last_interaction["net_credits"] == -5
    assert action(world, "wait")["reward"] == 0


def test_win_returns_stake_and_profit_exactly_once():
    world = World()
    world.roulette_rng = FixedOutcome(1)
    approach(world, "roulette")
    action(world, "bet_red")
    assert finish(world) == 5
    assert world.bank == 105
    action(world, "wait")
    assert world.bank == 105
    assert world.total_reward == 5


def test_empty_bank_stays_empty_until_explicit_refill():
    world = World()
    world.bank = 4
    approach(world, "roulette")
    result = action(world, "bet_black")
    assert result["event"] == "insufficient_credits"
    assert result["reward"] == -world.config["invalid_action_penalty"]
    assert world.bank == 4
    assert world.pending is None
    assert world.add_credits(10) == 14
    assert world.total_reward == -world.config["invalid_action_penalty"]
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            world.add_credits(invalid)


@pytest.mark.parametrize("sid", ["reels", "hookah", "whisky", "claude"])
def test_rewards_only_after_duration_and_fresh_availability(sid):
    world = World()
    approach(world, sid)
    assert action(world, "interact")["reward"] == 0
    assert world.total_reward == 0
    assert finish(world) == 1
    assert world.stations[sid]["uses"] == 1
    assert action(world, "interact")["reward"] == -world.config["invalid_action_penalty"]
    assert world.pending is None
    assert world.total_reward == 1 - world.config["invalid_action_penalty"]


def test_swiping_cannot_pay_without_completed_playback():
    world = World()
    approach(world, "reels")
    action(world, "interact")
    for _ in range(50):
        assert action(world, "swipe")["reward"] == 0
    assert world.total_reward == 0
    assert world.stations["reels"]["uses"] == 0
    assert finish(world) == 1


def test_interaction_requires_proximity_and_facing():
    world = World()
    assert action(world, "interact")["event"] == "nothing_in_reach"
    approach(world, "whisky")
    world.heading = math.pi
    assert action(world, "interact")["event"] == "nothing_in_reach"
    world.heading = 0
    assert action(world, "interact")["event"] == "interaction_started"
    before = world.x, world.z
    action(world, "forward")
    assert (world.x, world.z) == before


def test_interaction_rejects_station_center_outside_eye_field_of_view():
    world = World()
    approach(world, "whisky")
    world.heading = math.radians(40)
    assert action(world, "interact")["event"] == "nothing_in_reach"
    world.heading = math.radians(30)
    assert action(world, "interact")["event"] == "interaction_started"


def test_satiation_changes_reward_but_not_roulette_accounting():
    world = World()
    world.configure({"reward_mode": "satiation", "satiation_strength": 1, "satiation_recovery": 0})
    world.stations["roulette"]["satiation"] = 0.5
    world.roulette_rng = FixedOutcome(1)
    approach(world, "roulette")
    action(world, "bet_red")
    assert finish(world) == 2.5
    assert world.bank == 105


@pytest.mark.parametrize("bet,number", [("bet_red", 2), ("bet_black", 1), ("bet_red", 0), ("bet_black", 0)])
def test_roulette_loss_stays_negative_at_full_satiation(bet, number):
    world = World()
    world.configure({"reward_mode": "satiation", "satiation_strength": 1, "satiation_recovery": 0})
    world.stations["roulette"]["satiation"] = 1.0
    world.roulette_rng = FixedOutcome(number)
    approach(world, "roulette")
    action(world, bet)
    assert finish(world) == -5
    assert world.bank == 95


def test_roulette_win_stays_positive_at_full_satiation_unless_disabled():
    for scale in (1, 0):
        world = World()
        world.configure({"reward_mode": "satiation", "satiation_strength": 1, "satiation_recovery": 0,
                         "rewards": {"roulette": scale}})
        world.stations["roulette"]["satiation"] = 1.0
        world.roulette_rng = FixedOutcome(1)
        approach(world, "roulette")
        action(world, "bet_red")
        assert finish(world) == 0.25 * scale
        assert world.bank == 105


def test_configuration_is_validated_atomically():
    world = World()
    before = world.snapshot()["config"]
    for patch in ({"reward_mode": "addiction"}, {"rewards": {"whisky": 11}},
                  {"rewards": {"roulette": -0.1}},
                  {"rewards": {"unknown": 1}}, {"rewards": {"reels": float("nan")}},
                  {"navigation_shaping": 1}, {"satiation_strength": -1},
                  {"satiation_recovery": float("inf")}, {"bank": 10000},
                  {"invalid_action_penalty": -0.1}, {"invalid_action_penalty": 1.1},
                  {"invalid_action_penalty": True}, {"invalid_action_penalty": float("nan")},
                  {"reward_mode": "satiation", "rewards": {"reels": True}}):
        with pytest.raises(ValueError):
            world.configure(patch)
        assert world.snapshot()["config"] == before
    world.configure({"rewards": {"reels": 2}})
    assert world.config["rewards"]["reels"] == 2
    assert world.config["rewards"]["whisky"] == 1
    assert world.config["navigation_shaping"] is False


def test_render_is_deterministic_rgb_and_depends_on_pose():
    first, second = World(41), World(41)
    original = first.frame()
    assert original.shape == (48, 64, 3)
    assert original.dtype == np.uint8
    assert np.array_equal(original, second.frame())
    assert np.array_equal(original, first.frame())
    action(first, "turn_right")
    assert not np.array_equal(original, first.frame())
    first.heading = 0
    action(first, "forward")
    assert not np.array_equal(original, first.frame())
    json.dumps(first.snapshot(), allow_nan=False)


def test_rendering_does_not_change_roulette_random_sequence():
    first, second = World(82), World(82)
    first.frame()
    first.frame()
    for world in (first, second):
        approach(world, "roulette")
        action(world, "bet_red")
        finish(world)
    assert first.last_interaction["number"] == second.last_interaction["number"]


@pytest.mark.parametrize("sid", ["reels", "hookah", "whisky", "claude", "roulette"])
def test_station_state_is_visible_in_eye_pixels(sid):
    world = World()
    approach(world, sid)
    ready = world.frame()
    action(world, "bet_red" if sid == "roulette" else "interact")
    busy = world.frame()
    assert not np.array_equal(ready, busy)
    finish(world)
    unavailable = world.frame()
    assert not np.array_equal(ready, unavailable)
    assert not np.array_equal(busy, unavailable)


def test_roulette_bank_gauge_and_refill_change_eye_pixels():
    world = World()
    approach(world, "roulette")
    full = world.frame()
    world.bank = 95
    assert not np.array_equal(full, world.frame())
    world.bank = 5
    low = world.frame()
    assert not np.array_equal(full, low)
    world.bank = 0
    empty = world.frame()
    assert not np.array_equal(low, empty)
    world.add_credits(100)
    assert np.array_equal(full, world.frame())


def test_reels_animation_changes_during_playback_before_reward():
    world = World()
    approach(world, "reels")
    action(world, "interact")
    early = world.frame()
    for _ in range(4):
        assert action(world, "wait")["reward"] == 0
    assert not np.array_equal(early, world.frame())
    assert world.stations["reels"]["uses"] == 0


def test_renderer_respects_occlusion_and_field_of_view():
    world = World()
    near_box = (np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 0.0]), np.array([220, 30, 40]))
    hidden_sphere = (np.array([0.0, 0.75, -2.0]), 0.1, np.array([30, 230, 40]))
    world._geometry = lambda: ([near_box], [])
    foreground = world.frame()
    world._geometry = lambda: ([near_box], [hidden_sphere])
    assert np.array_equal(foreground, world.frame())
    world._geometry = lambda: ([], [hidden_sphere])
    assert not np.array_equal(foreground, world.frame())
    world._geometry = lambda: ([], [])
    empty_room = world.frame()
    behind_camera = (np.array([0.0, 0.75, 3.0]), 0.5, np.array([250, 250, 20]))
    world._geometry = lambda: ([], [behind_camera])
    assert np.array_equal(empty_room, world.frame())


def test_movement_stays_in_room_and_outside_station_bodies():
    world = World()
    world.x = 5
    for heading in (0, math.pi / 2, math.pi, -math.pi / 2):
        world.heading = heading
        for _ in range(100):
            action(world, "forward")
            assert abs(world.x) <= 6 - world.FLY_RADIUS
            assert abs(world.z) <= 6 - world.FLY_RADIUS
    approach(world, "whisky")
    for _ in range(10):
        action(world, "forward")
    station = world.stations["whisky"]
    assert math.hypot(world.x - station["x"], world.z - station["z"]) >= 0.68


def test_invalid_actions_do_not_advance_time():
    world = World()
    for value in (-1, len(ACTIONS), 1.1, True, "wait"):
        with pytest.raises(ValueError):
            world.step(value)
    assert world.time_s == 0


@pytest.mark.parametrize("sid,turn,turns,steps", [
    ("reels", "turn_left", 2, 12), ("hookah", "turn_right", 2, 12),
    ("whisky", "turn_left", 7, 9), ("claude", "turn_right", 7, 9),
    ("roulette", "turn_right", 0, 14),
])
def test_all_stations_reachable_from_normal_spawn(sid, turn, turns, steps):
    world = World()
    for _ in range(turns):
        action(world, turn)
    for _ in range(steps):
        action(world, "forward")
    result = action(world, "bet_red" if sid == "roulette" else "interact")
    assert result["event"] in ("interaction_started", "playback_started")
    assert world.pending["station"] == sid
    assert result["action_effective"] is True
    assert result["outcome_ready"] is False
    finish(world)
    assert world.stations[sid]["uses"] == 1


@pytest.mark.parametrize("sid", ["reels", "hookah", "whisky", "claude", "roulette"])
def test_pending_actions_are_not_credited_and_zero_reward_completes(sid):
    world = World()
    world.configure({"rewards": {sid: 0}})
    approach(world, sid)
    first = action(world, "bet_red" if sid == "roulette" else "interact")
    assert first["action_effective"] and not first["outcome_ready"]
    results = []
    while world.pending:
        results.append(action(world, "bet_black"))
    assert all(not result["action_effective"] for result in results)
    assert all(not result["outcome_ready"] for result in results[:-1])
    assert results[-1]["outcome_ready"]
    assert all(result["reward"] == 0 for result in results)


def test_active_reels_swipe_is_effective_but_defers_outcome():
    world = World()
    approach(world, "reels")
    action(world, "interact")
    previous_end = world.pending["ends_at"]
    result = action(world, "swipe")
    assert result["event"] == "clip_swiped"
    assert result["action_effective"] and not result["outcome_ready"]
    assert result["reward"] == 0
    assert world.pending["ends_at"] > previous_end


@pytest.mark.parametrize("penalty", [None, 0, 0.1, 1])
@pytest.mark.parametrize("sid,action_name,event", [
    (None, "interact", "nothing_in_reach"), ("hookah", "swipe", "not_at_phone"),
    ("whisky", "bet_red", "not_at_roulette"), ("roulette", "interact", "choose_red_or_black"),
    ("roulette", "bet_black", "insufficient_credits"), ("hookah", "interact", "cooldown"),
    ("claude", "interact", "request_unavailable"),
])
def test_invalid_attempt_penalty_is_explicit_and_configurable(penalty, sid, action_name, event):
    world = World()
    if penalty is not None:
        world.configure({"invalid_action_penalty": penalty})
    if sid:
        approach(world, sid)
    if event == "insufficient_credits":
        world.bank = 0
    if event in ("cooldown", "request_unavailable"):
        world.stations[sid]["available_at"] = 10
    result = action(world, action_name)
    assert result["event"] == event
    assert result["reward"] == -(0.02 if penalty is None else penalty)
    assert result["action_effective"] and result["outcome_ready"]


def test_failure_log_groups_only_consecutive_identical_failures():
    world = World()
    for _ in range(5):
        action(world, "interact")
    assert len(world.recent_events) == 1
    assert world.recent_events[-1] == {"time_s": 0.25, "last_time_s": 1.25, "count": 5,
                                      "event": "nothing_in_reach", "station": None, "reward": -0.1}
    action(world, "turn_right")
    action(world, "interact")
    assert len(world.recent_events) == 2
    assert world.recent_events[-1]["count"] == 1
    approach(world, "hookah")
    action(world, "bet_red")
    assert len(world.recent_events) == 3
    action(world, "interact")
    finish(world)
    assert [e["event"] for e in list(world.recent_events)[-2:]] == ["interaction_started", "hookah_used"]
    assert all("count" not in e for e in list(world.recent_events)[-2:])


def test_snapshot_roundtrip_preserves_pending_roulette_rng_and_history():
    world = World(82)
    world.configure({"rewards": {"reels": 2}, "invalid_action_penalty": 0.3})
    world.clip = 7
    approach(world, "roulette")
    action(world, "bet_red")
    finish(world)
    action(world, "wait")
    action(world, "wait")
    action(world, "bet_black")
    assert world.pending is not None
    saved = json.loads(json.dumps(world.snapshot(), allow_nan=False))
    restored = World.from_snapshot(saved)
    assert restored.snapshot() == saved
    for _ in range(30):
        assert action(world, "wait") == action(restored, "wait")
    assert restored.snapshot() == world.snapshot()
    assert np.array_equal(restored.frame(), world.frame())


def test_legacy_snapshot_retains_world_and_reconstructs_roulette_rng():
    world = World()
    approach(world, "roulette")
    action(world, "bet_red")
    finish(world)
    legacy = world.snapshot()
    for key in ("seed", "clip", "pending", "roulette_rng_state"):
        legacy.pop(key)
    legacy["config"].pop("invalid_action_penalty")
    restored = World.from_snapshot(legacy)
    assert restored.time_s == world.time_s
    assert restored.bank == world.bank
    assert restored.total_reward == world.total_reward
    assert restored.stations == world.stations
    assert restored.clip == 0
    assert restored.config["invalid_action_penalty"] == 0.02
    assert restored.roulette_rng.bit_generator.state == world.roulette_rng.bit_generator.state


@pytest.mark.parametrize("field,value", [("bank", -1), ("clip", True), ("time_s", float("nan"))])
def test_invalid_snapshot_rejected(field, value):
    saved = World().snapshot()
    saved[field] = value
    with pytest.raises(ValueError):
        World.from_snapshot(saved)


@pytest.mark.parametrize("sid", ["whisky", "roulette"])
def test_reset_position_cancels_activity_and_refunds_only_unsettled_bet(sid):
    world = World(15)
    world.total_reward = 10
    approach(world, sid)
    action(world, "bet_red" if sid == "roulette" else "interact")
    before = world.snapshot()
    event = world.reset_position()
    assert (world.x, world.z, world.heading) == (0, 1, 0)
    assert world.pending is None and world.last_interaction is None
    assert world.bank == 100
    assert world.time_s == before["time_s"] and world.total_reward == 10
    assert world.last_reward == 0 and world.last_event == "position_reset"
    assert event["refund_credits"] == (5 if sid == "roulette" else 0)
    assert event["reward"] == 0
    assert world.snapshot()["stations"] == [{**s, "available": True} for s in before["stations"]]
    world.reset_position()
    assert world.bank == 100


def test_reset_after_settled_bet_does_not_refund_or_reset_history():
    world = World()
    approach(world, "roulette")
    action(world, "bet_red")
    finish(world)
    before = world.snapshot()
    world.reset_position()
    assert world.bank == before["bank"]
    assert world.total_reward == before["total_reward"]
    assert world.time_s == before["time_s"]
    assert world.snapshot()["stations"] == before["stations"]
