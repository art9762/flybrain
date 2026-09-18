"""A small fictional room with pixel-only agent observations.

The station positions and interaction state in ``snapshot`` are for the human
interface and diagnostics. A controller should observe ``frame()`` only. Object
use is scripted; choosing an action and approaching a station are not scripted.
"""

from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np


ACTIONS = ("forward", "turn_left", "turn_right", "interact", "bet_red", "bet_black", "swipe", "wait")
STATIONS = (
    ("reels", "Reels", -3.0, -3.0, (253, 114, 154)),
    ("hookah", "Hookah", 3.0, -3.0, (185, 147, 255)),
    ("whisky", "Whisky", -3.0, 3.0, (242, 184, 87)),
    ("claude", "Claude Code", 3.0, 3.0, (142, 220, 194)),
    ("roulette", "Roulette", 0.0, -4.7, (116, 169, 255)),
)
RED_NUMBERS = frozenset((1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36))


class World:
    """Deterministic world; each step advances exactly 0.25 simulated seconds."""

    DT = 0.25
    BOUNDS = 6.0
    FLY_RADIUS = 0.18
    INTERACTION_RADIUS = 1.5
    FACING_TOLERANCE = math.radians(36)
    STAKE = 5
    DURATIONS = {"reels": 4.0, "hookah": 2.5, "whisky": 2.0, "claude": 0.5, "roulette": 3.0}
    COOLDOWNS = {"reels": 0.5, "hookah": 3.0, "whisky": 3.0, "claude": 5.0, "roulette": 0.5}
    INVALID_ACTION_EVENTS = frozenset(("nothing_in_reach", "not_at_phone", "not_at_roulette",
                                      "choose_red_or_black", "insufficient_credits", "cooldown",
                                      "request_unavailable"))

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.roulette_rng = np.random.default_rng(seed)
        self.time_s = 0.0
        self.x = 0.0
        self.z = 1.0
        self.heading = 0.0  # Heading zero faces negative z.
        self.bank = 100
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_event = "ready"
        self.last_interaction: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=40)
        self.config: dict[str, Any] = {
            "reward_mode": "constant",
            "rewards": {"reels": 1.0, "hookah": 1.0, "whisky": 1.0, "claude": 1.0, "roulette": 1.0},
            "satiation_strength": 0.8,
            "satiation_recovery": 0.005,
            "navigation_shaping": False,
            "invalid_action_penalty": 0.02,
        }
        self.stations = {
            sid: {"id": sid, "label": label, "x": x, "z": z, "color": list(color),
                  "satiation": 0.0, "uses": 0, "available_at": 0.0}
            for sid, label, x, z, color in STATIONS
        }
        self.clip = 0
        # Cached camera-plane coordinates, not a symbolic observation channel.
        sx = (2 * (np.arange(64) + 0.5) / 64 - 1) * math.tan(math.radians(72) / 2)
        sy = (1 - 2 * (np.arange(48) + 0.5) / 48) * math.tan(math.radians(72) / 2) * 48 / 64
        self._screen_x, self._screen_y = np.meshgrid(sx, sy)

    @staticmethod
    def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return float(value)

    def configure(self, patch: dict) -> dict:
        """Validate an entire patch before applying it; omitted values survive."""
        if not isinstance(patch, dict):
            raise ValueError("configuration must be an object")
        allowed = set(self.config)
        if set(patch) - allowed:
            raise ValueError(f"unknown configuration keys: {sorted(set(patch) - allowed)}")
        updated = {**self.config, "rewards": dict(self.config["rewards"])}
        if "reward_mode" in patch:
            if patch["reward_mode"] not in ("constant", "satiation"):
                raise ValueError("reward_mode must be constant or satiation")
            updated["reward_mode"] = patch["reward_mode"]
        if "rewards" in patch:
            rewards = patch["rewards"]
            if not isinstance(rewards, dict) or set(rewards) - set(self.stations):
                raise ValueError("rewards must map known station ids to numbers")
            for sid, value in rewards.items():
                minimum = 0 if sid == "roulette" else -10
                updated["rewards"][sid] = self._number(value, f"rewards.{sid}", minimum, 10)
        for key, maximum in (("satiation_strength", 1.0), ("satiation_recovery", 1.0),
                             ("invalid_action_penalty", 1.0)):
            if key in patch:
                updated[key] = self._number(patch[key], key, 0, maximum)
        if "navigation_shaping" in patch:
            if type(patch["navigation_shaping"]) is not bool:
                raise ValueError("navigation_shaping must be a boolean")
            updated["navigation_shaping"] = patch["navigation_shaping"]
        self.config = updated
        return {**updated, "rewards": dict(updated["rewards"])}

    def add_credits(self, amount: int) -> int:
        """Explicit human refill; adding credits never awards agent reward."""
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("credit amount must be a positive integer")
        self.bank += amount
        self.recent_events.append({"time_s": self.time_s, "event": "credits_added", "amount": amount, "reward": 0.0})
        return self.bank

    def reset_position(self) -> dict:
        """Return to spawn without erasing learning or completed world history."""
        refund = self.STAKE if self.pending and self.pending["station"] == "roulette" else 0
        self.bank += refund
        self.x, self.z, self.heading = 0.0, 1.0, 0.0
        self.pending = None
        self.last_interaction = None
        self.last_event = "position_reset"
        self.last_reward = 0.0
        event = {"time_s": self.time_s, "event": "position_reset", "reward": 0.0,
                 "refund_credits": refund}
        self.recent_events.append(event)
        return dict(event)

    def _near_station(self) -> str | None:
        candidates = []
        for sid, station in self.stations.items():
            dx, dz = station["x"] - self.x, station["z"] - self.z
            distance = math.hypot(dx, dz)
            angle = math.atan2(dx, -dz)
            difference = math.atan2(math.sin(angle - self.heading), math.cos(angle - self.heading))
            if distance <= self.INTERACTION_RADIUS and abs(difference) <= self.FACING_TOLERANCE:
                candidates.append((distance, sid))
        return min(candidates)[1] if candidates else None

    def _minimum_distance(self) -> float:
        return min(math.hypot(s["x"] - self.x, s["z"] - self.z) for s in self.stations.values())

    def _start(self, sid: str, bet: str | None = None) -> str:
        if self.time_s < self.stations[sid]["available_at"]:
            return "request_unavailable" if sid == "claude" else "cooldown"
        if sid == "roulette":
            if bet is None:
                return "choose_red_or_black"
            if self.bank < self.STAKE:
                return "insufficient_credits"
            self.bank -= self.STAKE
        self.pending = {"station": sid, "started_at": self.time_s,
                        "ends_at": self.time_s + self.DURATIONS[sid], "bet": bet}
        self.last_interaction = {**self.pending, "phase": "active", "progress": 0.0, "gesture": self._gesture(sid)}
        return "playback_started" if sid == "reels" else "interaction_started"

    @staticmethod
    def _gesture(sid: str) -> str:
        return {"reels": "watch_phone", "hookah": "hold_hose", "whisky": "lift_bottle",
                "claude": "press_confirm", "roulette": "place_chip"}[sid]

    def _complete(self) -> tuple[float, str, str]:
        assert self.pending is not None
        pending = self.pending
        sid = pending["station"]
        station = self.stations[sid]
        factor = 1.0
        if self.config["reward_mode"] == "satiation":
            factor = max(0.0, 1.0 - self.config["satiation_strength"] * station["satiation"])
        base = self.config["rewards"][sid]
        event = {"reels": "clip_watched", "hookah": "hookah_used", "whisky": "drink_finished",
                 "claude": "request_confirmed", "roulette": "roulette_resolved"}[sid]
        details: dict[str, Any] = {}
        if sid == "roulette":
            # No outcome is drawn at bet time or exposed in a pending snapshot.
            number = int(self.roulette_rng.integers(0, 37))
            color = "green" if number == 0 else "red" if number in RED_NUMBERS else "black"
            won = color == pending["bet"]
            net = self.STAKE if won else -self.STAKE
            if won:
                self.bank += 2 * self.STAKE
                # Enabled wins retain a positive signal at full satiation.
                factor = max(0.05, factor)
            else:
                # Roulette losses retain their penalty at full satiation.
                factor = 1.0
            base *= net
            event = "roulette_win" if won else "roulette_loss"
            details = {"number": number, "color": color, "won": won, "net_credits": net, "bet": pending["bet"]}
        reward = float(base * factor)
        station["uses"] += 1
        station["satiation"] = min(1.0, station["satiation"] + 0.25)
        station["available_at"] = self.time_s + self.COOLDOWNS[sid]
        self.last_interaction = {**pending, "phase": "complete", "progress": 1.0,
                                 "gesture": self._gesture(sid), "reward": reward, **details}
        self.pending = None
        return reward, event, sid

    def step(self, action_index: int) -> dict:
        if isinstance(action_index, bool) or not isinstance(action_index, (int, np.integer)) or not 0 <= action_index < len(ACTIONS):
            raise ValueError(f"action_index must be an integer from 0 to {len(ACTIONS) - 1}")
        action = ACTIONS[action_index]
        reward = 0.0
        sid: str | None = None
        event = action
        action_effective = True
        old_distance = self._minimum_distance()
        if self.pending is not None:
            action_effective = False
            sid = self.pending["station"]
            event = "busy"
            if sid == "reels" and action == "swipe":
                action_effective = True
                self.clip += 1
                self.pending["started_at"] = self.time_s
                self.pending["ends_at"] = self.time_s + self.DURATIONS["reels"]
                event = "clip_swiped"
        elif action == "forward":
            x = self.x + 0.32 * math.sin(self.heading)
            z = self.z - 0.32 * math.cos(self.heading)
            bound = self.BOUNDS - self.FLY_RADIUS
            x, z = max(-bound, min(bound, x)), max(-bound, min(bound, z))
            # Station bodies are obstacles. The agent can still stand within
            # interaction distance on any side and face the visible object.
            if all(math.hypot(s["x"] - x, s["z"] - z) >= 0.68 for s in self.stations.values()):
                self.x, self.z = x, z
            else:
                event = "obstacle"
        elif action in ("turn_left", "turn_right"):
            self.heading += math.radians(18) * (-1 if action == "turn_left" else 1)
            self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        elif action in ("interact", "bet_red", "bet_black", "swipe"):
            sid = self._near_station()
            if sid is None:
                event = "nothing_in_reach"
            elif action.startswith("bet_"):
                event = self._start(sid, action[4:]) if sid == "roulette" else "not_at_roulette"
            elif action == "swipe":
                if sid == "reels":
                    if self.time_s >= self.stations[sid]["available_at"]:
                        self.clip += 1
                    event = self._start(sid)
                else:
                    event = "not_at_phone"
            else:
                event = self._start(sid)
        if event in self.INVALID_ACTION_EVENTS:
            reward -= self.config["invalid_action_penalty"]
        if self.config["navigation_shaping"]:
            reward += 0.02 * (old_distance - self._minimum_distance())
        self.time_s += self.DT
        for station in self.stations.values():
            station["satiation"] = max(0.0, station["satiation"] - self.config["satiation_recovery"] * self.DT)
        if self.pending is not None:
            duration = self.pending["ends_at"] - self.pending["started_at"]
            progress = min(1.0, (self.time_s - self.pending["started_at"]) / duration)
            self.last_interaction = {**self.pending, "phase": "active", "progress": progress,
                                     "gesture": self._gesture(self.pending["station"])}
            if self.time_s >= self.pending["ends_at"]:
                earned, event, sid = self._complete()
                reward += earned
        self.last_reward = float(reward)
        self.total_reward += reward
        if event not in ("forward", "turn_left", "turn_right", "wait", "busy"):
            record = {"time_s": self.time_s, "event": event, "station": sid, "reward": float(reward)}
            if event in self.INVALID_ACTION_EVENTS:
                record.update(count=1, last_time_s=self.time_s)
                previous = self.recent_events[-1] if self.recent_events else None
                if (self.last_event == event and previous and previous["event"] == event
                        and previous.get("station") == sid):
                    record["time_s"] = previous["time_s"]
                    record["count"] = previous.get("count", 1) + 1
                    record["reward"] += previous["reward"]
                    self.recent_events.pop()
            self.recent_events.append(record)
        self.last_event = event
        return {"reward": float(reward), "event": event, "station": sid,
                "action_effective": action_effective, "outcome_ready": self.pending is None}

    def snapshot(self) -> dict:
        stations = []
        for station in self.stations.values():
            busy = self.pending is not None and self.pending["station"] == station["id"]
            stations.append({**station, "color": list(station["color"]),
                             "available": not busy and self.time_s >= station["available_at"]
                             and (station["id"] != "roulette" or self.bank >= self.STAKE)})
        return {"time_s": self.time_s, "fly": {"x": self.x, "z": self.z, "heading": self.heading},
                "seed": self.seed, "clip": self.clip, "pending": deepcopy(self.pending),
                "roulette_rng_state": deepcopy(self.roulette_rng.bit_generator.state)
                if hasattr(self.roulette_rng, "bit_generator") else None,
                "stations": stations, "bank": self.bank, "last_event": self.last_event,
                "last_reward": self.last_reward, "total_reward": self.total_reward,
                "config": {**self.config, "rewards": dict(self.config["rewards"])},
                "recent_events": [dict(event) for event in self.recent_events],
                "last_interaction": dict(self.last_interaction) if self.last_interaction else None,
                "renderer": {"width": 64, "height": 48, "fov_degrees": 72,
                             "camera_height": 0.75, "room_bounds": [-6, 6], "stake": self.STAKE}}

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> World:
        """Restore a local maintenance snapshot, including unfinished activity.

        Legacy snapshots have no clip/seed/RNG fields: clip defaults to zero,
        seed to 42, and the original generator is replayed once per settled bet.
        Legacy active interactions are reconstructed from last_interaction.
        """
        if not isinstance(snapshot, dict):
            raise ValueError("world snapshot must be an object")

        def integer(value, name):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            return int(value)

        world = cls(integer(snapshot.get("seed", 42), "seed"))
        world.configure(snapshot.get("config", {}))
        world.time_s = world._number(snapshot["time_s"], "time_s", 0, math.inf)
        pose = snapshot["fly"]
        bound = cls.BOUNDS - cls.FLY_RADIUS
        world.x = world._number(pose["x"], "fly.x", -bound, bound)
        world.z = world._number(pose["z"], "fly.z", -bound, bound)
        world.heading = world._number(pose["heading"], "fly.heading", -math.pi, math.pi)
        world.bank = integer(snapshot["bank"], "bank")
        world.clip = integer(snapshot.get("clip", 0), "clip")
        for key in ("last_reward", "total_reward"):
            setattr(world, key, world._number(snapshot[key], key, -math.inf, math.inf))
        stations = snapshot["stations"]
        if (not isinstance(stations, list) or len(stations) != len(world.stations)
                or any(not isinstance(station, dict) for station in stations)
                or {s.get("id") for s in stations} != set(world.stations)):
            raise ValueError("snapshot must contain every station exactly once")
        for saved in stations:
            station = world.stations[saved["id"]]
            station["uses"] = integer(saved["uses"], "station.uses")
            station["satiation"] = world._number(saved["satiation"], "station.satiation", 0, 1)
            station["available_at"] = world._number(saved["available_at"], "station.available_at", 0, math.inf)
        world.last_event = str(snapshot.get("last_event", "ready"))
        world.last_interaction = deepcopy(snapshot.get("last_interaction"))
        if world.last_interaction is not None and not isinstance(world.last_interaction, dict):
            raise ValueError("last_interaction must be an object or null")
        pending = deepcopy(snapshot.get("pending"))
        if "pending" not in snapshot and world.last_interaction and world.last_interaction.get("phase") == "active":
            pending = {key: world.last_interaction[key] for key in ("station", "started_at", "ends_at", "bet")}
        if pending is not None:
            if not isinstance(pending, dict) or pending.get("station") not in world.stations:
                raise ValueError("invalid pending station")
            start = world._number(pending["started_at"], "pending.started_at", 0, world.time_s)
            end = world._number(pending["ends_at"], "pending.ends_at", world.time_s, math.inf)
            if end <= start or (pending["station"] == "roulette" and pending.get("bet") not in ("red", "black")):
                raise ValueError("invalid pending activity")
            world.pending = {"station": pending["station"], "started_at": start, "ends_at": end,
                             "bet": pending.get("bet")}
        recent = snapshot.get("recent_events", [])
        if not isinstance(recent, list) or any(not isinstance(e, dict) for e in recent):
            raise ValueError("recent_events must be an array of event objects")
        world.recent_events = deque(deepcopy(recent[-40:]), maxlen=40)
        rng_state = snapshot.get("roulette_rng_state")
        if rng_state is not None:
            try:
                world.roulette_rng.bit_generator.state = deepcopy(rng_state)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("invalid roulette RNG state") from error
        else:
            for _ in range(world.stations["roulette"]["uses"]):
                world.roulette_rng.integers(0, 37)
        return world

    def _geometry(self) -> tuple[list, list]:
        """World-space solid boxes and spheres; no projected station icons."""
        boxes, spheres = [], []

        def box(center, size, color):
            center, half = np.array(center), np.array(size) / 2
            boxes.append((center - half, center + half, np.array(color)))

        def sphere(center, radius, color):
            spheres.append((np.array(center), radius, np.array(color)))

        for sid, station in self.stations.items():
            x, z, color = station["x"], station["z"], station["color"]
            box((x, 0.12, z), (1.2, 0.24, 1.2), [int(c * 0.45) for c in color])
            box((x, 0.44, z), (0.16, 0.65, 0.16), (66, 71, 85))
            box((x, 0.75, z), (1.1, 0.1, 0.8), color)
            busy = self.pending is not None and self.pending["station"] == sid
            available = self.time_s >= station["available_at"] and (sid != "roulette" or self.bank >= self.STAKE)
            status = (247, 203, 80) if busy else (53, 220, 126) if available else (230, 71, 86)
            box((x + 0.45, 0.9, z + 0.28), (0.055, 0.2, 0.055), (64, 70, 83))
            sphere((x + 0.45, 1.04, z + 0.28), 0.075, status)
            if sid == "reels":
                box((x, 1.17, z), (0.57, 0.79, 0.14), (16, 22, 35))
                playing = self.pending is not None and self.pending["station"] == "reels"
                elapsed = self.time_s - self.pending["started_at"] if playing else 0.0
                phase = elapsed * 2.2 + self.clip * 1.37
                # A deliberately abstract but changing frame: colored bars
                # and a moving subject are rendered as world geometry.
                palette = ((253, 114, 154), (105, 168, 255), (247, 207, 118), (96, 208, 153))
                for i, dz in enumerate((-0.078, 0.078)):
                    direction = -1 if dz < 0 else 1
                    box((x, 1.18, z + dz), (0.47, 0.65, 0.015), palette[self.clip % len(palette)])
                    for band in range(3):
                        shift = 0.07 * math.sin(phase + band * 1.6)
                        box((x + shift, 0.99 + band * 0.17, z + dz + direction * 0.015),
                            (0.31, 0.085, 0.01), palette[(self.clip + band + 1) % len(palette)])
                    offset = 0.14 * math.sin(phase + i * 1.9)
                    sphere((x + offset, 1.18 + 0.12 * math.cos(phase + i), z + dz + direction * 0.025), 0.075, (248, 231, 172))
            elif sid == "hookah":
                sphere((x, 0.96, z), 0.19, color)
                box((x, 1.21, z), (0.065, 0.43, 0.065), (209, 207, 213))
                sphere((x, 1.46, z), 0.11, (210, 119, 94))
                for t in np.linspace(0, 1, 7):
                    sphere((x + 0.42 * t, 1.16 - 0.29 * math.sin(t * math.pi), z + 0.12 * t), 0.04, (71, 47, 91))
            elif sid == "whisky":
                box((x, 1.01, z), (0.29, 0.45, 0.22), (174, 99, 37))
                box((x, 1.30, z), (0.10, 0.20, 0.10), (204, 154, 70))
                box((x, 1.43, z), (0.13, 0.07, 0.13), (36, 38, 37))
                box((x, 1.02, z + 0.116), (0.23, 0.19, 0.02), (235, 217, 176))
            elif sid == "claude":
                box((x, 0.85, z + 0.13), (0.72, 0.07, 0.45), (91, 96, 111))
                box((x, 1.12, z - 0.08), (0.79, 0.51, 0.06), (25, 32, 40))
                for dz in (-0.116, -0.044):
                    box((x - 0.07, 1.18, z + dz), (0.50, 0.035, 0.016), (202, 172, 154))
                    box((x, 1.03, z + dz), (0.38, 0.10, 0.016), status)
            else:
                # A fictional online casino display, not a physical wheel.
                # Both screen faces are visible so either approach is usable.
                box((x, 1.08, z), (1.05, 0.60, 0.08), (18, 27, 40))
                phase = self.time_s * 3 if busy else 0.0
                for side in (-1, 1):
                    screen_z = z + side * 0.05
                    box((x, 1.08, screen_z), (0.95, 0.50, 0.015), (27, 71, 62))
                    for i in range(12):
                        angle = i * math.tau / 12 + phase
                        sphere((x - 0.21 + 0.15 * math.sin(angle), 1.13 + 0.15 * math.cos(angle),
                                screen_z + side * 0.025), 0.031,
                               (222, 63, 71) if i % 2 else (29, 32, 41))
                    sphere((x - 0.21, 1.13, screen_z + side * 0.03), 0.055, (217, 189, 110))
                    box((x + 0.25, 1.19, screen_z + side * 0.025), (0.23, 0.12, 0.02), (222, 63, 71))
                    box((x + 0.25, 0.99, screen_z + side * 0.025), (0.23, 0.12, 0.02), (29, 32, 41))
                # A physical balance panel stands on each side of the table.
                # Fill represents 0..100 credits; an extra lit cap marks >100.
                for side in (-1, 1):
                    panel_z = z + side * 0.36
                    box((x, 0.95, panel_z), (0.80, 0.18, 0.045), (22, 31, 46))
                    box((x, 0.95, panel_z + side * 0.028), (0.72, 0.10, 0.02), (74, 83, 103))
                    fill = min(1.0, self.bank / 100)
                    if fill > 0:
                        box((x - 0.36 + 0.36 * fill, 0.95, panel_z + side * 0.043),
                            (0.72 * fill, 0.10, 0.012), (53, 220, 126))
                    sphere((x + 0.36, 1.075, panel_z), 0.035,
                           (249, 213, 100) if self.bank > 100 else (62, 69, 83))
        return boxes, spheres

    def frame(self) -> np.ndarray:
        """Render a 48x64 uint8 RGB perspective image with depth occlusion."""
        origin = np.array((self.x, 0.75, self.z))
        forward = np.array((math.sin(self.heading), 0.0, -math.cos(self.heading)))
        right = np.array((math.cos(self.heading), 0.0, math.sin(self.heading)))
        rays = (forward + self._screen_x[..., None] * right
                + self._screen_y[..., None] * np.array((0.0, 1.0, 0.0))).reshape(-1, 3)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        depth = np.full(len(rays), np.inf)
        rgb = np.zeros((len(rays), 3), dtype=np.float64)
        # Six room planes establish the distant background. The room is closed
        # and the camera remains inside, so nearest-plane selection is exact.
        surfaces = ((0, -6, (45, 51, 67)), (0, 6, (57, 61, 77)),
                    (2, -6, (55, 59, 79)), (2, 6, (46, 50, 64)),
                    (1, 0, (72, 75, 84)), (1, 4, (36, 39, 52)))
        for axis, coordinate, color in surfaces:
            distance = np.divide(coordinate - origin[axis], rays[:, axis],
                                 out=np.full(len(rays), np.inf), where=np.abs(rays[:, axis]) > 1e-12)
            hit = (distance > 0) & (distance < depth)
            depth[hit] = distance[hit]
            rgb[hit] = color
            if axis == 1 and coordinate == 0 and np.any(hit):
                points = origin + rays[hit] * distance[hit, None]
                tiles = (np.floor(points[:, 0]) + np.floor(points[:, 2])).astype(int) % 2
                rgb[hit] *= np.where(tiles[:, None] == 0, 0.86, 1.05)
        boxes, spheres = self._geometry()
        parallel = np.abs(rays) < 1e-12
        safe_rays = np.where(parallel, 1.0, rays)
        for lower, upper, color in boxes:
            a = (lower - origin) / safe_rays
            b = (upper - origin) / safe_rays
            near = np.where(parallel, -np.inf, np.minimum(a, b))
            far = np.where(parallel, np.inf, np.maximum(a, b))
            misses = np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
            entry, leave = np.max(near, axis=1), np.min(far, axis=1)
            distance = np.where(entry > 1e-6, entry, leave)
            hit = ~misses & (leave >= entry) & (distance > 1e-6) & (distance < depth)
            if np.any(hit):
                face_axis = np.argmax(near[hit], axis=1)
                shade = np.choose(face_axis, (0.83, 1.0, 0.93))
                rgb[hit] = color * shade[:, None]
                depth[hit] = distance[hit]
        light = np.array((-0.35, 0.85, 0.40))
        light /= np.linalg.norm(light)
        for center, radius, color in spheres:
            oc = origin - center
            b = np.sum(rays * oc, axis=1)
            discriminant = b * b - (np.sum(oc * oc) - radius * radius)
            root = np.sqrt(np.maximum(discriminant, 0))
            distance = -b - root
            distance = np.where(distance > 1e-6, distance, -b + root)
            hit = (discriminant >= 0) & (distance > 1e-6) & (distance < depth)
            if np.any(hit):
                normals = (origin + rays[hit] * distance[hit, None] - center) / radius
                shade = 0.45 + 0.55 * np.maximum(0, np.sum(normals * light, axis=1))
                rgb[hit] = color * shade[:, None]
                depth[hit] = distance[hit]
        # Mild depth attenuation preserves object colors at long distance.
        rgb *= (0.70 + 0.30 * np.exp(-depth / 12))[:, None]
        return np.clip(rgb, 0, 255).astype(np.uint8).reshape(48, 64, 3)
