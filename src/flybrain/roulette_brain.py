"""Numeric roulette interface to the complete FlyWire recurrent controller.

The fixed encoder repeats 48 normalized public observations into the existing
768-feature projection. The topology, neural dynamics and motor populations are
unchanged. Both rewards and teacher targets modify only magnitudes of existing
connectome edges into the eight motor populations. This is an artificial rate
model on a biological graph, not a biological simulation of a fly playing cards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import expit

from .brain import Brain
from .connectome import Connectome


ACTIONS = (
    "shoot_opponent", "shoot_self", "magnifier", "beer",
    "cigarette", "saw", "cuffs", "inverter",
)
OBSERVATION_SIZE = 48
OBSERVATION_SCHEMA = "roulette-public-numeric-48-v1"
ACTION_SCHEMA = "roulette-eight-actions-v1"


class RouletteBrain(Brain):
    """Full-graph policy with explicit numeric input and legal-action masking.

    ``act(observation, learn=False, legal_mask=...)`` samples a legal action and
    requires exactly one subsequent ``feedback`` call, like the base controller.
    ``teach`` is one complete, independent supervised transition; it resets
    neural state and traces, does not sample an action, and leaves no feedback
    pending. Its returned loss, chosen action and target probability describe
    the policy *before* the update. The inherited learning-update counter counts
    both supervised and reward updates. Teacher labels never enter the encoder.
    """

    action_schema = ACTION_SCHEMA
    observation_schema = OBSERVATION_SCHEMA

    def __init__(
        self,
        data_dir: Path,
        seed: int = 42,
        *,
        graph: Connectome | None = None,
        backend: str = "auto",
    ) -> None:
        self._legal_mask = np.ones(len(ACTIONS), dtype=bool)
        super().__init__(data_dir, seed=seed, graph=graph, backend=backend)

    @staticmethod
    def encode_vision(observation: np.ndarray) -> np.ndarray:
        """Fixed direct numeric projection; there is no intermediate image."""
        values = np.asarray(observation)
        if values.shape != (OBSERVATION_SIZE,):
            raise ValueError("roulette observation must have shape (48,)")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("roulette observation must contain floating-point numbers")
        if not np.isfinite(values).all() or np.any(values < -1) or np.any(values > 1):
            raise ValueError("roulette observation must be finite and between -1 and 1")
        return np.tile(values.astype(np.float32, copy=False), 16)

    @staticmethod
    def _validate_mask(legal_mask: np.ndarray | None) -> np.ndarray:
        if legal_mask is None:
            return np.ones(len(ACTIONS), dtype=bool)
        mask = np.asarray(legal_mask)
        if mask.shape != (len(ACTIONS),) or mask.dtype != np.bool_:
            raise ValueError("legal_mask must be a boolean vector with shape (8,)")
        if not mask.any():
            raise ValueError("legal_mask must permit at least one action")
        return mask.copy()

    def _probabilities(self, state: np.ndarray) -> np.ndarray:
        logits = self.POLICY_SCALE * state[self.motor_neurons].mean(axis=1).astype(np.float64)
        logits[~self._legal_mask] = -np.inf
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def act(
        self, observation: np.ndarray, learn: bool = False, *,
        legal_mask: np.ndarray | None = None,
    ) -> int:
        if self._pending:
            raise RuntimeError("call feedback once before requesting another action")
        # Validate before changing any controller state.
        self.encode_vision(observation)
        self._legal_mask = self._validate_mask(legal_mask)
        return super().act(observation, learn=learn)

    def teach(
        self, observation: np.ndarray, target_distribution: np.ndarray, *,
        legal_mask: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        """Minimize cross entropy with a teacher, using only existing edges.

        The derivative detaches the state before the final recurrent step,
        matching the base brain's reward-learning approximation. Independent
        samples have no temporal trace. Reward baseline and RNG are untouched.
        Every sample still evaluates every graph edge in all recurrent steps.
        """
        if self._pending:
            raise RuntimeError("call feedback before teaching a new observation")
        features = self.encode_vision(observation)
        mask = self._validate_mask(legal_mask)
        target = np.asarray(target_distribution, dtype=np.float64)
        if (target.shape != (len(ACTIONS),) or not np.isfinite(target).all()
                or np.any(target < 0) or not np.isclose(target.sum(), 1.0, atol=1e-8, rtol=0)):
            raise ValueError("target_distribution must be eight finite nonnegative probabilities summing to one")
        if np.any(target[~mask] > 0):
            raise ValueError("target_distribution cannot assign probability to an illegal action")

        self.reset_activity()
        self._legal_mask = mask
        drive = np.sum(features[self._visual_indices] * self._visual_weights, axis=1)
        for _ in range(self.RECURRENT_STEPS):
            previous = self.state
            rates = expit(self.operator.matvec(2 * previous - 1) + drive + self.BIAS)
            self.state = ((1 - self.LEAK) * previous + self.LEAK * rates).astype(np.float32)
        probabilities = self._probabilities(self.state)
        self.action_probabilities = probabilities
        supported = target > 0
        loss = -float(np.dot(target[supported], np.log(np.maximum(probabilities[supported], 1e-300))))
        chosen = int(np.argmax(probabilities))
        # E_teacher[d log pi(a)] is exactly the negative cross-entropy gradient.
        for action in np.flatnonzero(supported):
            self.eligibility += float(target[action]) * self._local_score(
                int(action), previous, rates, probabilities
            )
        self._apply_reward_update(1.0)
        self.eligibility.fill(0)
        self.learning_updates += 1
        self.frames += 1
        self.neural_steps += self.RECURRENT_STEPS
        self.last_action = chosen
        # Supervised targets are not game rewards and cannot be reinforced by
        # a subsequent manual reward. The next ordinary act/feedback still works.
        self._can_reinforce = False
        return {"loss": loss, "chosen": chosen, "targetprob": float(np.dot(target, probabilities))}

    def reset_activity(self) -> None:
        super().reset_activity()
        self._legal_mask.fill(True)

    def _model_parameters(self) -> dict:
        return {
            **super()._model_parameters(),
            "actionSchema": self.action_schema,
            "actions": list(ACTIONS),
            "observationSchema": self.observation_schema,
            "observationSize": OBSERVATION_SIZE,
            "observationRange": [-1, 1],
            "numericProjection": "repeat-48-features-16-times-v1",
            "legalActionMask": True,
        }

    def observe_status(self) -> dict:
        return {
            **super().observe_status(),
            "actions": list(ACTIONS),
            "actionSchema": self.action_schema,
            "observationSchema": self.observation_schema,
            "policyInput": "48 normalized public game values plus legal-action mask",
            "learningRule": "teacher cross entropy or local recurrent policy gradient on existing signed edges",
        }
