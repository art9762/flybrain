"""One local simulation session. All world/brain mutations are serialized."""

from __future__ import annotations

import base64
from collections import deque
from pathlib import Path
import threading
import time

from .world import ACTIONS, World


class LabSession:
    def __init__(self, root: Path):
        self.root = root
        self.world = World(seed=42)
        self.brain = None
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.worker = None
        self.running = False
        self.learning = True
        self.pending_steps = 0
        self.backend = "auto"
        self.speed = 1.0
        self.loading = False
        self.error = None
        self.tick_count = 0
        self.last_action = None
        self.last_effective_action = None
        self.last_step_ms = 0.0
        self.last_frame = self.world.frame()
        self.reward_history = deque(maxlen=240)
        self.manual_total = 0.0
        self.manual_events = deque(maxlen=30)
        self.recent_activity = deque(maxlen=240)

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.loading = self.brain is None
            self.worker = threading.Thread(target=self._loop, name="flybrain-loop", daemon=True)
            self.worker.start()

    def control(self, action: str, learning: bool | None = None,
                speed: float | None = None, backend: str | None = None) -> dict:
        with self.condition:
            if action not in ("start", "pause", "step"):
                raise ValueError("Неизвестная команда")
            if backend is not None and backend != self.backend:
                if self.brain is not None or self.loading:
                    raise ValueError("Для смены CPU/GPU перезапустите сервер; обученные веса можно сохранить.")
                self.backend = backend
            if learning is not None:
                self.learning = learning
            if speed is not None:
                self.speed = speed
            if action == "pause":
                self.running = False
                self.pending_steps = 0
            elif action in ("start", "step"):
                self.error = None
                self.running = action == "start"
                if action == "step":
                    self.pending_steps += 1
                self._ensure_worker()
            else:
                raise ValueError("Неизвестная команда")
            self.condition.notify_all()
            return self.status()

    def _loop(self) -> None:
        try:
            if self.brain is None:
                with self.lock:
                    self.loading = True
                from .brain import Brain
                brain = Brain(data_dir=self.root / "data", seed=42, backend=self.backend)
                with self.lock:
                    self.brain = brain
                    self.loading = False
            while True:
                with self.condition:
                    while not self.running and self.pending_steps == 0:
                        self.condition.wait()
                    if self.pending_steps:
                        self.pending_steps -= 1
                    started = time.perf_counter()
                    # The controller receives only the eye pixels, never this world's snapshot.
                    observation = self.world.frame()
                    action = self.brain.act(observation, learn=self.learning)
                    result = self.world.step(action)
                    reward = float(result["reward"])
                    self.brain.feedback(reward, learn=self.learning,
                                        action_effective=result["action_effective"],
                                        outcome_ready=result["outcome_ready"])
                    self.last_frame = observation
                    self.last_action = ACTIONS[action] if result["action_effective"] else "busy"
                    if result["action_effective"]:
                        self.last_effective_action = ACTIONS[action]
                    self.recent_activity.append({"action": self.last_action, "event": result["event"]})
                    self.tick_count += 1
                    self.last_step_ms = (time.perf_counter() - started) * 1000
                    self.reward_history.append({"step": self.tick_count, "reward": reward})
                    delay = max(0.0, 0.25 / self.speed - self.last_step_ms / 1000)
                    # Condition.wait releases the world lock and permits immediate pause.
                    self.condition.wait(timeout=delay)
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
                self.loading = False
                self.running = False
                self.pending_steps = 0

    def status(self) -> dict:
        with self.lock:
            return {
                "world": self.world.snapshot(),
                "brain": self.brain.observe_status() if self.brain is not None else None,
                "run": {"running": self.running, "loading": self.loading,
                        "learning": self.learning, "speed": self.speed,
                        "backend_requested": self.backend, "steps": self.tick_count,
                        "last_action": self.last_action,
                        "last_effective_action": self.last_effective_action,
                        "step_ms": round(self.last_step_ms, 2),
                        "error": self.error, "manual_reward": self.manual_total},
                "eye": {"width": int(self.last_frame.shape[1]),
                        "height": int(self.last_frame.shape[0]),
                        "rgb": base64.b64encode(self.last_frame.tobytes()).decode("ascii")},
                "rewards": list(self.reward_history),
                "manual_events": list(self.manual_events),
                "activity": {
                    "window_steps": len(self.recent_activity),
                    "movement_steps": sum(a["action"] in ("forward", "turn_left", "turn_right")
                                          for a in self.recent_activity),
                    "invalid_attempts": sum(a["event"] in World.INVALID_ACTION_EVENTS
                                            for a in self.recent_activity),
                    "completed_uses": sum(s["uses"] for s in self.world.stations.values()),
                },
            }

    def reset_position(self) -> dict:
        """Pause and relocate only the body; retain learned weights and rewards."""
        with self.condition:
            if self.loading:
                raise ValueError("Дождитесь загрузки мозга перед возвратом в центр")
            self.running = False
            self.pending_steps = 0
            self.world.reset_position()
            if self.brain is not None:
                # Stale movement/eligibility must not attach to the new pose.
                self.brain.reset_activity()
            self.last_frame = self.world.frame()
            self.last_action = None
            self.last_effective_action = None
            self.recent_activity.clear()
            self.condition.notify_all()
            return self.status()

    def configure(self, patch: dict) -> dict:
        with self.lock:
            self.world.configure(patch)
            return self.status()

    def reward(self, amount: float) -> dict:
        with self.lock:
            if self.brain is None or self.tick_count == 0:
                raise ValueError("Сначала запустите хотя бы один шаг мухи")
            if not self.learning:
                raise ValueError("Ручное подкрепление доступно при включённом обучении")
            if not self.brain.observe_status().get("canReinforce", False):
                raise ValueError("Сделайте шаг с включённым обучением перед ручной наградой")
            self.brain.reinforce(amount)
            self.manual_total += amount
            self.manual_events.append({"step": self.tick_count, "reward": amount,
                                       "action": self.last_effective_action, "source": "human"})
            self.reward_history.append({"step": self.tick_count, "reward": amount, "source": "human"})
            return self.status()

    def credits(self, amount: int) -> dict:
        with self.lock:
            self.world.add_credits(amount)
            return self.status()

    def checkpoint(self, action: str) -> dict:
        with self.lock:
            if self.brain is None:
                raise ValueError("Сначала загрузите мозг кнопкой «Один шаг»")
            path = self.root / "outputs" / "brain-latest.npz"
            if action == "save":
                path.parent.mkdir(parents=True, exist_ok=True)
                self.brain.save(path)
            elif action == "load":
                if self.running:
                    raise ValueError("Поставьте мир на паузу перед загрузкой")
                if not path.is_file():
                    raise ValueError("Сохранения ещё нет")
                self.brain.load(path)
            else:
                raise ValueError("Неизвестное действие с сохранением")
            return {"path": str(path), "action": action, "scope": "brain_only", "ok": True}
