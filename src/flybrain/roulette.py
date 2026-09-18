"""Headless, information-safe shotgun duel for the connectome experiment.

This is an original implementation of the live/blank turn mechanic. A saw is
consumed by the user's next shot *or beer ejection*. Cuffs skip one hand-off,
and each player may apply them once per magazine. A public inverter changes
the next shell without revealing its type. Public counts therefore describe
the magazine before the pending inversion until that shell is removed.

Neither observation(), snapshot(), nor teacher_action() reads hidden order.
Only the action resolver and magazine loader inspect the actual shell queue.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np


ACTIONS = (
    "shoot_opponent", "shoot_self", "magnifier", "beer", "cigarette", "saw",
    "cuffs", "inverter",
)
ITEMS = ACTIONS[2:]
OBSERVATION_SIZE = 48
MAX_HP = 4
ITEM_LABELS = {
    "magnifier": "лупу", "beer": "пиво", "cigarette": "сигарету",
    "saw": "пилу", "cuffs": "наручники", "inverter": "инвертор",
}


class RouletteGame:
    """A seeded two-player duel; player 0 is human and player 1 is the fly.

    Invalid actions raise ValueError without changing any state. ``step``
    returns a public event, including actor-relative ``reward`` and zero-sum
    ``rewards`` for both players. Inspection results are available only through
    the inspector's snapshot/observation, never through shared events.
    """

    def __init__(
        self, seed: int = 42, starting_player: int = 0, items: bool = True,
        max_actions: int = 256,
    ) -> None:
        self._check_player(starting_player)
        if isinstance(max_actions, bool) or not isinstance(max_actions, Integral) or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.turn = int(starting_player)
        self.winner: int | None = None
        self.hp = [MAX_HP, MAX_HP]
        self.items_enabled = bool(items)
        self.max_actions = int(max_actions)
        self.action_count = 0
        self.load_number = 0
        self.inventories = [{item: 0 for item in ITEMS} for _ in range(2)]
        self.saw_active = [False, False]
        self.cuffed = [False, False]
        self.cuffs_used = [False, False]
        self._known_shells: list[bool | None] = [None, None]
        self._shells: list[bool] = []
        self._public_live = 0
        self._public_blank = 0
        self._inverted = False
        self._last_action: int | None = None
        self.history: list[dict[str, Any]] = []
        self._reload()

    @staticmethod
    def _check_player(player: int) -> None:
        if isinstance(player, bool) or not isinstance(player, Integral) or player not in (0, 1):
            raise ValueError("player must be 0 or 1")

    @property
    def shells_remaining(self) -> int:
        return self._public_live + self._public_blank

    @property
    def events(self) -> list[dict[str, Any]]:
        """Copy of the public history, including load events."""
        return deepcopy(self.history)

    def _reload(self) -> None:
        count = int(self.rng.integers(2, 9))
        live = int(self.rng.integers(1, count))
        self._shells = [True] * live + [False] * (count - live)
        self.rng.shuffle(self._shells)
        self._public_live, self._public_blank = live, count - live
        self._inverted = False
        self._known_shells = [None, None]
        self.cuffs_used = [False, False]
        self.load_number += 1
        granted: list[list[str]] = [[], []]
        if self.items_enabled:
            for player in (0, 1):
                for _ in range(min(2, 8 - sum(self.inventories[player].values()))):
                    item = ITEMS[int(self.rng.integers(len(ITEMS)))]
                    self.inventories[player][item] += 1
                    granted[player].append(item)
        self.history.append({
            "event": "reload", "load_number": self.load_number,
            "live": live, "blank": count - live, "granted": granted,
            "text": f"Зарядка {self.load_number}: боевых — {live}, холостых — {count - live}.",
        })

    def legal_actions(self, player: int | None = None) -> list[int]:
        player = self.turn if player is None else player
        self._check_player(player)
        if self.winner is not None or player != self.turn:
            return []
        legal = [0, 1]
        for action, item in enumerate(ITEMS, 2):
            if self.inventories[player][item] <= 0:
                continue
            if item == "magnifier" and self._known_shells[player] is not None:
                continue
            if item == "cigarette" and self.hp[player] >= MAX_HP:
                continue
            if item == "saw" and self.saw_active[player]:
                continue
            if item == "cuffs" and (self.cuffs_used[player] or self.cuffed[1 - player]):
                continue
            legal.append(action)
        return legal

    def visible_live_probability(self, player: int) -> float:
        """Posterior probability based solely on this player's information."""
        self._check_player(player)
        known = self._known_shells[player]
        if known is not None:
            return float(known)
        if self.shells_remaining == 0:
            return 0.5
        baseline = self._public_live / self.shells_remaining
        return 1.0 - baseline if self._inverted else baseline

    def _remove_shell(self) -> bool:
        live = self._shells.pop(0)
        original_live = not live if self._inverted else live
        if original_live:
            self._public_live -= 1
        else:
            self._public_blank -= 1
        self._inverted = False
        self._known_shells = [None, None]
        return live

    def _hand_off(self, actor: int, event: dict[str, Any]) -> None:
        other = 1 - actor
        if self.cuffed[other]:
            self.cuffed[other] = False
            event["skipped_player"] = other
            event["text"] += " Наручники пропускают ход соперника."
        else:
            self.turn = other

    def step(self, action: int) -> dict[str, Any]:
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action!r}")
        action = int(action)
        actor, other = self.turn, 1 - self.turn
        name = ACTIONS[action]
        who = "Игрок" if actor == 0 else "Муха"
        event: dict[str, Any] = {
            "event": name, "action": name, "action_index": action,
            "actor": actor, "text": "", "damage": 0,
        }
        reward = 0.0
        self.action_count += 1
        self._last_action = action
        if action < 2:
            live = self._remove_shell()
            target = other if action == 0 else actor
            damage = (2 if self.saw_active[actor] else 1) if live else 0
            self.saw_active[actor] = False
            self.hp[target] = max(0, self.hp[target] - damage)
            event.update(shell="live" if live else "blank", target=target, damage=damage)
            target_text = "в соперника" if action == 0 else "в себя"
            shell_text = "боевой" if live else "холостой"
            event["text"] = f"{who} стреляет {target_text}: {shell_text}."
            reward = damage / MAX_HP * (1 if target == other else -1)
            if self.hp[target] == 0:
                self.winner = 1 - target
                reward += 1 if self.winner == actor else -1
                event["text"] += " Дуэль завершена."
            elif action == 1 and not live:
                event["text"] += " Ход сохраняется."
            else:
                self._hand_off(actor, event)
        else:
            self.inventories[actor][name] -= 1
            event["text"] = f"{who} использует {ITEM_LABELS[name]}."
            if name == "magnifier":
                self._known_shells[actor] = self._shells[0]
                event["text"] += " Результат видит только владелец лупы."
            elif name == "beer":
                live = self._remove_shell()
                self.saw_active[actor] = False
                event["shell"] = "live" if live else "blank"
                event["text"] += " Извлечён " + ("боевой." if live else "холостой.")
            elif name == "cigarette":
                self.hp[actor] += 1
                reward = 1 / MAX_HP
                event["healed"] = 1
                event["text"] += " Восстановлена 1 единица здоровья."
            elif name == "saw":
                self.saw_active[actor] = True
                event["text"] += " Следующий выстрел наносит двойной урон."
            elif name == "cuffs":
                self.cuffed[other] = True
                self.cuffs_used[actor] = True
                event["text"] += " Следующая передача хода сопернику будет пропущена."
            elif name == "inverter":
                self._shells[0] = not self._shells[0]
                self._inverted = not self._inverted
                self._known_shells = [None if known is None else not known for known in self._known_shells]
                event["text"] += " Следующий патрон изменён без раскрытия типа."

        if self.winner is None and self.action_count >= self.max_actions:
            self.winner = -1
            event["text"] += " Лимит действий: ничья."
        rewards = [0.0, 0.0]
        rewards[actor], rewards[other] = reward, -reward
        reloaded = self.winner is None and self.shells_remaining == 0
        event.update(
            reward=reward, rewards=rewards, winner=self.winner,
            turn=self.turn, reloaded=reloaded, action_count=self.action_count,
        )
        self.history.append(deepcopy(event))
        if reloaded:
            self._reload()
        return deepcopy(event)

    def observation(self, player: int) -> np.ndarray:
        """48 float32 values; layout is stable and documented below.

        0..19: own/opponent HP, own turn, terminal, own outcome,
        remaining shells, baseline live/blank counts, visible live probability,
        privately known shell (-1 blank, 0 unknown, +1 live), own/other saw,
        own/other cuffs, own/other cuffs-used, inversion, exact-counts flag,
        load progress, action budget progress.
        20..25 and 26..31: own/opponent item counts in ITEMS order.
        32..39: own legal action mask; 40..47: previous public action one-hot.
        HP/count/progress inputs are normalized to [0,1].
        """
        self._check_player(player)
        other = 1 - player
        known = self._known_shells[player]
        outcome = 0 if self.winner is None or self.winner == -1 else (1 if self.winner == player else -1)
        obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        obs[:20] = (
            self.hp[player] / MAX_HP, self.hp[other] / MAX_HP,
            float(self.turn == player), float(self.winner is not None), outcome,
            self.shells_remaining / 8, self._public_live / 8, self._public_blank / 8,
            self.visible_live_probability(player), 0 if known is None else (1 if known else -1),
            self.saw_active[player], self.saw_active[other], self.cuffed[player], self.cuffed[other],
            self.cuffs_used[player], self.cuffs_used[other], self._inverted, not self._inverted,
            min(self.load_number / 20, 1), self.action_count / self.max_actions,
        )
        obs[20:26] = [self.inventories[player][item] / 8 for item in ITEMS]
        obs[26:32] = [self.inventories[other][item] / 8 for item in ITEMS]
        obs[[32 + action for action in self.legal_actions(player)]] = 1
        if self._last_action is not None:
            obs[40 + self._last_action] = 1
        return obs

    def snapshot(self, player: int = 0) -> dict[str, Any]:
        """JSON-ready view for one player; never exposes hidden shell order."""
        self._check_player(player)
        known = self._known_shells[player]
        return {
            "viewer": player, "turn": self.turn, "winner": self.winner,
            "hp": list(self.hp), "max_hp": MAX_HP, "load_number": self.load_number,
            "shells_remaining": self.shells_remaining,
            "live": None if self._inverted else self._public_live,
            "blank": None if self._inverted else self._public_blank,
            "counts_exact": not self._inverted, "inverted": self._inverted,
            "counts_before_inversion": {"live": self._public_live, "blank": self._public_blank},
            "inventories": deepcopy(self.inventories), "saw_active": list(self.saw_active),
            "cuffed": list(self.cuffed), "cuffs_used": list(self.cuffs_used),
            "known_shell": None if known is None else ("live" if known else "blank"),
            "legal_actions": self.legal_actions(player), "log": deepcopy(self.history),
            "action_count": self.action_count, "max_actions": self.max_actions,
        }


def teacher_action(game: RouletteGame, player: int | None = None) -> int:
    """A deterministic heuristic teacher restricted to the player's view.

    This teaches information use and item combinations, not optimal play.
    Its decisions are invariant to permutations of unknown shell order.
    """
    player = game.turn if player is None else player
    legal = game.legal_actions(player)
    if not legal:
        raise ValueError("The player has no legal action")
    view = game.snapshot(player)
    p_live = game.visible_live_probability(player)
    other = 1 - player
    damage = 2 if view["saw_active"][player] else 1

    # Take certain wins immediately; otherwise restore health while able.
    if p_live == 1 and view["hp"][other] <= damage:
        return 0
    if 4 in legal:
        return 4
    if 2 in legal and 0 < p_live < 1:
        return 2
    if p_live == 0:
        if 7 in legal:
            return 7
        return 1
    if p_live == 1:
        if 5 in legal and view["hp"][other] > 1:
            return 5
        if 6 in legal and view["shells_remaining"] > 1 and view["hp"][other] > damage:
            return 6
        return 0
    if 7 in legal and p_live < 0.5:
        return 7
    if 6 in legal and view["shells_remaining"] > 1:
        return 6
    if 3 in legal and view["hp"][player] <= 1 and p_live < 0.5:
        return 3
    if 5 in legal and p_live >= 0.5:
        return 5
    return 0 if p_live >= 0.5 or view["hp"][player] <= 1 else 1
