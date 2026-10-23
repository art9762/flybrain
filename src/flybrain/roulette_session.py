"""Serialized game actions and a bounded background connectome trainer."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading

from .roulette import RouletteGame
from .roulette_training import mask_for, train
from .roulette_reports import idle_training_status, load_training_status, save_training_report


class RouletteSession:
    def __init__(self, root, brain_factory=None):
        self.root = Path(root)
        self.brain_factory = brain_factory
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.brain = None
        self.brain_status = None
        self.game = RouletteGame(seed=42)
        self.game_seed = 42
        self.loading = self.busy = self.training = False
        self.error = None
        self.backend = "auto"
        self.training_status = idle_training_status()
        self.worker = None

    @property
    def checkpoint_path(self):
        return self.root / "outputs" / "roulette-latest.npz"

    def status(self):
        with self.lock:
            return {"game": self.game.snapshot(0), "brain": self.brain_status,
                    "run": {"loading": self.loading, "busy": self.busy, "training": self.training,
                            "backend": self.backend, "error": self.error},
                    "training": dict(self.training_status),
                    "checkpoint_available": self.checkpoint_path.is_file()}

    def _idle(self):
        if self.loading or self.busy or self.training:
            raise ValueError("Дождитесь завершения текущей операции")

    def _brain_ready(self):
        self._idle()
        if self.brain is None:
            raise ValueError("Сначала разбудите муху")

    @staticmethod
    def _model_status(brain):
        full = brain.observe_status()
        # Action probabilities and neural activations could disclose a private
        # magnifier result. The opponent sees only model/training metadata.
        fields = ("neurons", "edges", "fullGraph", "plasticEdges", "changedWeights",
                  "learningUpdates", "backend", "seed", "graphChecksum", "model",
                  "actionSchema", "observationSchema", "policyInput", "learningRule")
        return {key: full[key] for key in fields if key in full}

    def _cache(self):
        self.brain_status = self._model_status(self.brain)

    def load_brain(self, backend="auto"):
        with self.lock:
            self._idle()
            if self.brain is not None:
                return self.status()
            self.loading, self.backend, self.error = True, backend, None
            self.worker = threading.Thread(target=self._load, daemon=True)
            self.worker.start()
            return self.status()

    def _load(self):
        try:
            from .roulette_brain import RouletteBrain
            factory = self.brain_factory or RouletteBrain
            brain = factory(self.root / "data", seed=42, backend=self.backend)
            if self.brain_factory is None and (brain.neuron_count != 138639 or brain.edge_count != 15091983):
                raise ValueError("Нужен полный коннектом FlyWire v783")
            if self.checkpoint_path.is_file():
                brain.load(self.checkpoint_path)
                brain.reset_activity()
                training_status = load_training_status(self.checkpoint_path.with_suffix(".json"), brain)
            else:
                training_status = idle_training_status()
            brain_status = self._model_status(brain)
            with self.lock:
                self.brain = brain
                self.brain_status = brain_status
                self.training_status = training_status
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            with self.lock:
                self.loading = False

    def new_game(self, starting_player=0):
        with self.lock:
            self._idle()
            self.game_seed += 1
            self.game = RouletteGame(seed=self.game_seed, starting_player=starting_player)
            self.error = None
            if self.brain:
                self.brain.reset_activity()
            return self.status()

    def action(self, action):
        with self.lock:
            self._brain_ready()
            if self.game.winner is not None or self.game.turn != 0:
                raise ValueError("Сейчас нельзя сделать ход игрока")
            self.game.step(action)
            return self.status()

    def fly(self):
        with self.lock:
            self._brain_ready()
            if self.game.winner is not None or self.game.turn != 1:
                raise ValueError("Сейчас не ход мухи")
            self.busy = True
            try:
                self.brain.reset_activity()
                action = self.brain.act(self.game.observation(1), learn=False,
                                        legal_mask=mask_for(self.game, 1))
                self.brain.feedback(0, learn=False)
                self.game.step(action)
                self._cache()
            finally:
                self.busy = False
            return self.status()

    def start_training(self, steps=2000):
        with self.lock:
            self._brain_ready()
            self.training = True
            self.error = None
            self.stop_event.clear()
            self.training_status = {"state": "training", "steps": 0, "target_steps": steps,
                                    "episodes": 0, "message": "Готовим обучение"}
            self.worker = threading.Thread(target=self._train, args=(steps,), daemon=True)
            self.worker.start()
            return self.status()

    def _train(self, steps):
        try:
            path = self.root / "outputs" / ("roulette-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            def progress(value):
                with self.lock:
                    self.training_status.update(value)
            report = train(self.brain, path, steps=steps, stopped=self.stop_event.is_set, progress=progress)
            self.brain.save(self.checkpoint_path)
            save_training_report(self.checkpoint_path.with_suffix(".json"), report)
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
                self.training_status.update(state="error", message=self.error)
        finally:
            with self.lock:
                self._cache()
                self.training = False

    def stop(self):
        self.stop_event.set()
        return self.status()

    def checkpoint(self, action):
        with self.lock:
            self._brain_ready()
            if action == "save":
                self.brain.save(self.checkpoint_path)
            elif action == "load":
                self.brain.load(self.checkpoint_path)
                self.brain.reset_activity()
                self.training_status = load_training_status(self.checkpoint_path.with_suffix(".json"), self.brain)
            else:
                raise ValueError("Неизвестное действие с сохранением")
            self._cache()
            return self.status()
