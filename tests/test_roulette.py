from copy import deepcopy
import json

import numpy as np
import pytest

from flybrain.roulette import ACTIONS, ITEMS, RouletteGame, teacher_action


def scenario(shells, *, turn=0, max_actions=256):
    game = RouletteGame(42, starting_player=turn, items=False, max_actions=max_actions)
    game._shells = list(shells)
    game._public_live = sum(shells)
    game._public_blank = len(shells) - sum(shells)
    game.history = []
    return game


def use(game, action):
    return game.step(ACTIONS.index(action))


def give(game, item, count=1, player=None):
    game.inventories[game.turn if player is None else player][item] += count


def test_seeded_loads_items_and_serializable_views():
    first, second = RouletteGame(17), RouletteGame(17)
    assert first.snapshot() == second.snapshot()
    assert first._shells == second._shells
    assert 2 <= first.shells_remaining <= 8
    assert first.snapshot()["live"] >= 1 and first.snapshot()["blank"] >= 1
    assert all(sum(items.values()) == 2 for items in first.inventories)
    json.dumps(first.snapshot())
    assert first.observation(0).shape == (48,)
    assert first.observation(0).dtype == np.float32


def test_hidden_order_and_opponent_inspection_do_not_change_player_view_or_teacher():
    first = scenario([True, False, False, True])
    give(first, "magnifier")
    second = deepcopy(first)
    second._shells.reverse()
    second._shells[0], second._shells[1] = second._shells[1], second._shells[0]
    assert first._shells[0] != second._shells[0]
    assert first.snapshot() == second.snapshot()
    np.testing.assert_array_equal(first.observation(0), second.observation(0))
    assert teacher_action(first) == teacher_action(second)
    assert use(first, "magnifier") == use(second, "magnifier")
    assert first.history == second.history
    assert first.snapshot(0)["known_shell"] != second.snapshot(0)["known_shell"]
    assert first.snapshot(1) == second.snapshot(1)
    np.testing.assert_array_equal(first.observation(1), second.observation(1))


def test_magnifier_result_is_private_in_event_and_history():
    game = scenario([True, False])
    give(game, "magnifier")
    event = use(game, "magnifier")
    assert "shell" not in event and "known_shell" not in event
    assert game.snapshot(0)["known_shell"] == "live"
    assert game.snapshot(1)["known_shell"] is None
    assert game.visible_live_probability(0) == 1
    assert game.visible_live_probability(1) == 0.5
    assert 2 not in game.legal_actions()


@pytest.mark.parametrize("shell,target,expected_hp,expected_turn", [
    (False, "shoot_self", [4, 4], 0),
    (True, "shoot_self", [3, 4], 1),
    (False, "shoot_opponent", [4, 4], 1),
    (True, "shoot_opponent", [4, 3], 1),
])
def test_shot_damage_and_turn_rules(shell, target, expected_hp, expected_turn):
    game = scenario([shell, not shell])
    event = use(game, target)
    assert game.hp == expected_hp
    assert game.turn == expected_turn
    assert event["damage"] == int(shell)
    assert game.shells_remaining == 1


def test_saw_doubles_one_shot_and_is_consumed_by_blank_or_ejection():
    for shell in (True, False):
        game = scenario([shell, not shell])
        give(game, "saw")
        use(game, "saw")
        assert game.saw_active[0]
        assert 5 not in game.legal_actions()
        use(game, "shoot_opponent")
        assert game.hp[1] == (2 if shell else 4)
        assert not game.saw_active[0]
    game = scenario([True, False])
    give(game, "saw")
    give(game, "beer")
    use(game, "saw")
    event = use(game, "beer")
    assert event["shell"] == "live"
    assert game.hp == [4, 4] and game.turn == 0
    assert not game.saw_active[0]


def test_cuffs_skip_once_and_can_be_used_once_per_load():
    game = scenario([False, True, False, True])
    give(game, "cuffs", 2)
    use(game, "cuffs")
    assert game.cuffed == [False, True]
    use(game, "shoot_self")
    assert game.cuffed[1] and game.turn == 0  # A free shot does not consume cuffs.
    event = use(game, "shoot_opponent")
    assert game.turn == 0 and not game.cuffed[1]
    assert event["skipped_player"] == 1
    assert 6 not in game.legal_actions()
    use(game, "shoot_opponent")
    assert game.turn == 1


def test_healing_is_capped_and_only_consumes_when_legal():
    game = scenario([True, False])
    give(game, "cigarette", 2)
    with pytest.raises(ValueError):
        use(game, "cigarette")
    assert game.inventories[0]["cigarette"] == 2
    game.hp[0] = 3
    use(game, "cigarette")
    assert game.hp[0] == 4 and game.inventories[0]["cigarette"] == 1
    assert 4 not in game.legal_actions()


def test_inverter_conceals_changed_counts_and_preserves_fair_inference():
    game = scenario([True, False, False])
    give(game, "inverter", 2)
    use(game, "inverter")
    view = game.snapshot()
    assert view["live"] is None and view["blank"] is None
    assert view["counts_before_inversion"] == {"live": 1, "blank": 2}
    assert game.visible_live_probability(0) == pytest.approx(2 / 3)
    same_public = deepcopy(game)
    same_public._shells = [True, True, False]  # The opposite hidden baseline head.
    assert same_public.snapshot() == view
    np.testing.assert_array_equal(game.observation(0), same_public.observation(0))
    assert teacher_action(game) == teacher_action(same_public)
    event = use(game, "shoot_self")
    assert event["shell"] == "blank" and game.turn == 0
    assert game.snapshot()["live"] == 0 and game.snapshot()["blank"] == 2
    assert game.visible_live_probability(0) == 0


def test_inverter_updates_each_existing_private_inspection():
    game = scenario([True, False])
    game._known_shells = [True, True]
    give(game, "inverter", 2)
    use(game, "inverter")
    assert [game.snapshot(p)["known_shell"] for p in (0, 1)] == ["blank", "blank"]
    use(game, "inverter")
    assert game.snapshot()["counts_exact"]
    assert [game.visible_live_probability(p) for p in (0, 1)] == [1, 1]


def test_reload_invalidates_inspection_resets_cuffs_and_caps_inventory():
    game = scenario([False])
    game.items_enabled = True
    game.inventories[0] = {item: 0 for item in ITEMS}
    game.inventories[0]["beer"] = 7
    game.cuffs_used = [True, True]
    game._known_shells = [False, False]
    old_load = game.load_number
    event = use(game, "shoot_self")
    assert event["reloaded"] and game.load_number == old_load + 1
    assert game.turn == 0 and 2 <= game.shells_remaining <= 8
    assert game._known_shells == [None, None] and game.cuffs_used == [False, False]
    assert sum(game.inventories[0].values()) == 8
    assert sum(game.inventories[1].values()) == 2


@pytest.mark.parametrize("invalid", [-1, 8, 1.5, "shoot_self", True, None, 2])
def test_invalid_actions_do_not_mutate_state_or_rng(invalid):
    game = RouletteGame(1, items=False)
    before = deepcopy(game.__dict__)
    rng_before = deepcopy(game.rng.bit_generator.state)
    with pytest.raises(ValueError):
        game.step(invalid)
    after = dict(game.__dict__)
    after.pop("rng")
    before.pop("rng")
    assert before == after
    assert game.rng.bit_generator.state == rng_before


def test_winner_draw_and_terminal_action_rejection():
    game = scenario([True, False])
    game.hp[1] = 1
    event = use(game, "shoot_opponent")
    assert game.winner == 0 and event["winner"] == 0
    assert event["reward"] > 0 and sum(event["rewards"]) == 0
    assert game.legal_actions() == []
    assert game.observation(1)[4] == -1
    with pytest.raises(ValueError):
        use(game, "shoot_self")
    game = scenario([False, True], max_actions=1)
    use(game, "shoot_self")
    assert game.winner == -1 and game.legal_actions() == []
    assert game.observation(0)[4] == 0


def test_teacher_rollouts_end_and_observations_stay_bounded():
    for seed in range(80):
        game = RouletteGame(seed, starting_player=seed % 2)
        while game.winner is None:
            action = teacher_action(game)
            assert action in game.legal_actions()
            game.step(action)
            for player in (0, 1):
                obs = game.observation(player)
                assert np.all(np.isfinite(obs))
                assert np.all(obs >= -1) and np.all(obs <= 1)
                assert sum(game.inventories[player].values()) <= 8
        assert game.action_count <= game.max_actions


def test_snapshot_mutation_cannot_change_engine():
    game = RouletteGame()
    snapshot = game.snapshot()
    snapshot["hp"][0] = -500
    snapshot["inventories"][0]["beer"] = 99
    snapshot["log"][0]["live"] = 999
    assert game.hp[0] == 4
    assert game.inventories[0]["beer"] < 99
    assert game.history[0]["live"] < 999
