"""Full-connectome rate dynamics with local, reward-modulated plasticity.

This is an artificial rate approximation, not a biological LIF emulator. Pixel
projections and motor populations are synthetic interfaces. The actor changes
only magnitudes of existing recurrent edges into the motor populations; no
external trainable action readout is used. The local policy derivative treats
the state before the final recurrent step as constant (truncated recurrence),
then an eligibility trace carries that derivative to later rewards. The scalar
critic estimates mean reward, not a calibrated prediction of biological value.
Rates lie in [0, 1], while recurrent edges transmit the centered deviation
``2 * rate - 1``. Neutral rate 0.5 therefore contributes no tonic drive, preserving
visual differences that uncentered fan-in averaging would otherwise obscure.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit

from .connectome import Connectome, load_connectome
from .gpu import SparseOperator


ACTIONS = ("forward", "left", "right", "interact", "bet_red", "bet_black", "swipe", "wait")


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class Brain:
    """All provided neurons and edges participate in every recurrent step.

    ``act`` accepts only a 48x64 RGB image. Call ``feedback`` exactly once after
    each action. Evaluation remains stochastic but freezes all learned weights,
    the critic and eligibility traces; the neural state still evolves. For a
    matched replay restore a checkpoint, which includes neural and RNG state.
    ``graph`` is an injection seam for small tests, never a production sampler.
    """

    VERSION = 2
    RECURRENT_STEPS = 3
    RECURRENT_GAIN = 0.9
    LEAK = 0.65
    BIAS = 0.0
    POLICY_SCALE = 48.0
    VISUAL_GAIN = 4.0
    LEARNING_RATE = 4.0
    BASELINE_RATE = 0.02
    TRACE_DECAY = 0.94
    MIN_LOG_MULTIPLIER = float(np.log(0.1))
    MAX_LOG_MULTIPLIER = float(np.log(3.0))
    MAX_LOG_UPDATE = 0.08

    def __init__(
        self,
        data_dir: Path,
        seed: int = 42,
        *,
        graph: Connectome | None = None,
        backend: str = "auto",
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.data_source = "flywire-v783" if graph is None else "injected-test-graph"
        graph = load_connectome(Path(data_dir), minimum_synapses=1) if graph is None else graph
        self._validate_graph(graph)
        self.neuron_count = graph.neuron_count
        self.edge_count = graph.edge_count
        self.neuron_ids = graph.neuron_ids.copy()
        self.graph_checksum = _array_digest(
            {"ids": graph.neuron_ids, "indptr": graph.indptr,
             "targets": graph.targets, "weights": graph.weights}
        )

        # Convert outgoing CSR once. Transposition retains every stored edge.
        outgoing = sparse.csr_matrix(
            (np.asarray(graph.weights, dtype=np.float32), graph.targets, graph.indptr),
            shape=(self.neuron_count, self.neuron_count),
        )
        self.matrix = outgoing.transpose().tocsr(copy=True)
        del outgoing
        row_counts = np.diff(self.matrix.indptr)
        signed_strength = np.sign(self.matrix.data) * np.sqrt(np.abs(self.matrix.data))
        self.matrix.data[:] = np.abs(signed_strength)
        incoming_mass = np.asarray(self.matrix.sum(axis=1), dtype=np.float32).reshape(-1)
        incoming_mass = np.maximum(incoming_mass, 1.0)
        self.matrix.data[:] = signed_strength
        self.matrix.data *= np.repeat(self.RECURRENT_GAIN / incoming_mass, row_counts)
        del signed_strength, incoming_mass

        candidates = np.flatnonzero(row_counts > 0)
        if candidates.size < len(ACTIONS):
            raise ValueError("the graph needs at least eight neurons with incoming edges")
        self.motor_group_size = min(8, max(1, candidates.size // (2 * len(ACTIONS))))
        self.motor_neurons = self.rng.choice(
            candidates, len(ACTIONS) * self.motor_group_size, replace=False
        ).reshape(len(ACTIONS), self.motor_group_size)
        motor_flat = self.motor_neurons.reshape(-1)
        self.plastic_indices = np.concatenate(
            [np.arange(self.matrix.indptr[n], self.matrix.indptr[n + 1], dtype=np.int64)
             for n in motor_flat]
        )
        self.plastic_targets = np.repeat(motor_flat, row_counts[motor_flat])
        self.plastic_sources = self.matrix.indices[self.plastic_indices].copy()
        self._plastic_actions = np.repeat(
            np.repeat(np.arange(len(ACTIONS)), self.motor_group_size), row_counts[motor_flat]
        )
        self._base_plastic_weights = self.matrix.data[self.plastic_indices].copy()
        self.log_multipliers = np.zeros(self.plastic_indices.size, dtype=np.float32)
        self.eligibility = np.zeros_like(self.log_multipliers)
        self._pending_gradient = np.zeros_like(self.log_multipliers)

        # A fixed sparse projection: three pixel features per neuron. It uses
        # neither object labels nor world state. Motor cells have no direct
        # visual drive, so their visual responses pass through recurrent edges.
        self._visual_indices = self.rng.integers(0, 768, (self.neuron_count, 3), dtype=np.int16)
        self._visual_weights = self.rng.normal(
            0.0, self.VISUAL_GAIN / np.sqrt(3.0), (self.neuron_count, 3)
        ).astype(np.float32)
        self._visual_weights[motor_flat] = 0
        self._layout_checksum = _array_digest(
            {"visual_indices": self._visual_indices, "visual_weights": self._visual_weights,
             "motor_neurons": self.motor_neurons, "plastic_indices": self.plastic_indices}
        )
        self.operator = SparseOperator(self.matrix, backend=backend)
        self.state = np.full(self.neuron_count, 0.5, dtype=np.float32)
        self.action_probabilities = np.full(len(ACTIONS), 1 / len(ACTIONS), dtype=np.float64)
        self.frames = 0
        self.neural_steps = 0
        self.learning_updates = 0
        self.manual_reinforcements = 0
        self.manual_reward_total = 0.0
        self.baseline = 0.0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_advantage = 0.0
        self.last_action = -1
        self._pending = False
        self._pending_learn = False
        self._can_reinforce = False

    @staticmethod
    def _validate_graph(graph: Connectome) -> None:
        n, e = graph.neuron_count, graph.edge_count
        if n < 8 or e < 1:
            raise ValueError("the graph must contain at least eight neurons and one edge")
        if graph.indptr.shape != (n + 1,) or graph.indptr[0] != 0 or graph.indptr[-1] != e:
            raise ValueError("invalid graph row pointers")
        if np.any(np.diff(graph.indptr) < 0):
            raise ValueError("graph row pointers must be monotonic")
        if graph.weights.shape != (e,) or np.any(graph.targets < 0) or np.any(graph.targets >= n):
            raise ValueError("invalid graph edge arrays")
        if not np.all(np.isfinite(graph.weights)) or np.any(graph.weights == 0):
            raise ValueError("graph weights must be finite, nonzero and signed")
        if np.unique(graph.neuron_ids).size != n:
            raise ValueError("neuron IDs must be unique")

    @staticmethod
    def encode_vision(image: np.ndarray) -> np.ndarray:
        """Pool RGB into luminance, color-opponent and contrast features only."""
        image = np.asarray(image)
        if image.shape != (48, 64, 3):
            raise ValueError("vision must have shape (48, 64, 3)")
        if image.dtype == np.uint8:
            rgb = image.astype(np.float32) / 255.0
        elif np.issubdtype(image.dtype, np.floating):
            if not np.all(np.isfinite(image)) or image.min() < 0 or image.max() > 1:
                raise ValueError("floating RGB pixels must be finite and between zero and one")
            rgb = image.astype(np.float32, copy=False)
        else:
            raise ValueError("vision must be uint8 RGB or normalized floating RGB")
        pooled = rgb.reshape(12, 4, 16, 4, 3).mean(axis=(1, 3))
        r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.stack(
            (luminance, r - g, b - (r + g) * 0.5, luminance - luminance.mean()),
            axis=-1,
        ).reshape(-1).astype(np.float32, copy=False)

    def _probabilities(self, state: np.ndarray) -> np.ndarray:
        logits = self.POLICY_SCALE * state[self.motor_neurons].mean(axis=1).astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def _local_score(
        self, action: int, previous: np.ndarray, rates: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray:
        """Exact final-step d log pi(a) / d log(|w|), with previous detached."""
        dlogit = -probabilities.copy()
        dlogit[action] += 1.0
        scale = self.POLICY_SCALE * self.LEAK / self.motor_group_size
        target_rates = rates[self.plastic_targets]
        return (
            scale * dlogit[self._plastic_actions] * target_rates * (1 - target_rates)
            * (2 * previous[self.plastic_sources] - 1) * self.matrix.data[self.plastic_indices]
        ).astype(np.float32)

    def act(self, image: np.ndarray, learn: bool = True) -> int:
        if self._pending:
            raise RuntimeError("call feedback once before requesting another action")
        features = self.encode_vision(image)
        drive = np.sum(features[self._visual_indices] * self._visual_weights, axis=1)
        for _ in range(self.RECURRENT_STEPS):
            previous = self.state
            rates = expit(self.operator.matvec(2 * previous - 1) + drive + self.BIAS)
            self.state = ((1 - self.LEAK) * previous + self.LEAK * rates).astype(np.float32)
        self.action_probabilities = self._probabilities(self.state)
        action = int(self.rng.choice(len(ACTIONS), p=self.action_probabilities))
        self._pending_learn = bool(learn)
        if not learn:
            self._can_reinforce = False
        if learn:
            self._pending_gradient[:] = self._local_score(action, previous, rates, self.action_probabilities)
        self._pending = True
        self.last_action = action
        self.frames += 1
        self.neural_steps += self.RECURRENT_STEPS
        return action

    def feedback(
        self, reward: float, learn: bool = True, *,
        action_effective: bool = True, outcome_ready: bool = True,
    ) -> None:
        """Account for a tick and learn only from choices the world executed.

        Scripted activities may ignore sampled choices while the whole brain
        still runs on every eye frame. Those ticks retain the initiating trace
        without decay or unrelated gradients. A completed result, including
        zero reward, learns from that retained trace. A real choice during an
        activity (for example a swipe) contributes its own gradient normally.
        The flags describe feedback timing, not symbolic policy observations.
        Defaults preserve the original one-choice/one-outcome API and v2 saves.
        """
        if not self._pending:
            raise RuntimeError("feedback requires a preceding action")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.last_reward = reward
        self.total_reward += reward
        if learn and self._pending_learn:
            # A scalar average-reward critic provides the baseline. Eligibility
            # approximates temporal credit; there is no backpropagation through
            # earlier recurrent states. Projected log updates preserve all signs.
            if action_effective:
                self.eligibility *= self.TRACE_DECAY
                self.eligibility += self._pending_gradient
                self._can_reinforce = True
            if outcome_ready and self._can_reinforce:
                advantage = reward - self.baseline
                self.last_advantage = advantage
                self.baseline += self.BASELINE_RATE * advantage
                self._apply_reward_update(advantage)
                self.learning_updates += 1
        else:
            self._can_reinforce = False
        self._pending = False
        self._pending_learn = False

    def _apply_reward_update(self, advantage: float) -> None:
        update = np.clip(
            self.LEARNING_RATE * advantage * self.eligibility,
            -self.MAX_LOG_UPDATE, self.MAX_LOG_UPDATE,
        )
        self.log_multipliers += update
        np.clip(
            self.log_multipliers, self.MIN_LOG_MULTIPLIER, self.MAX_LOG_MULTIPLIER,
            out=self.log_multipliers,
        )
        values = self._base_plastic_weights * np.exp(self.log_multipliers)
        self.matrix.data[self.plastic_indices] = values
        self.operator.update_weights(self.plastic_indices, values)

    def reinforce(self, amount: float) -> None:
        """Add human reward to the most recent completed training transition.

        This reuses its retained eligibility trace, without another neural
        step, action sample, or trace decay. It is an incremental correction to
        the previous reward: update the actor with ``amount * eligibility`` and
        the critic with ``BASELINE_RATE * amount`` (no second baseline subtraction).
        Per-update clipping can differ from receiving the summed reward once.
        Repeated clicks are cumulative. Evaluation/reset clears this permission.
        """
        amount = float(amount)
        if not np.isfinite(amount):
            raise ValueError("manual reward must be finite")
        if self._pending or not self._can_reinforce:
            raise RuntimeError("manual reward requires a completed training action")
        self._apply_reward_update(amount)
        self.baseline += self.BASELINE_RATE * amount
        self.last_reward += amount
        self.last_advantage += amount
        self.total_reward += amount
        self.manual_reward_total += amount
        self.manual_reinforcements += 1

    def reset_activity(self) -> None:
        """Start a new episode while retaining weights and the reward baseline."""
        self.state.fill(0.5)
        self.eligibility.fill(0)
        self._pending_gradient.fill(0)
        self._pending = False
        self._pending_learn = False
        self._can_reinforce = False
        self.action_probabilities.fill(1 / len(ACTIONS))

    def _model_parameters(self) -> dict:
        return {
            "recurrentSteps": self.RECURRENT_STEPS, "recurrentGain": self.RECURRENT_GAIN,
            "leak": self.LEAK, "bias": self.BIAS, "policyScale": self.POLICY_SCALE,
            "learningRate": self.LEARNING_RATE, "baselineRate": self.BASELINE_RATE,
            "traceDecay": self.TRACE_DECAY, "maxLogUpdate": self.MAX_LOG_UPDATE,
            "minLogMultiplier": self.MIN_LOG_MULTIPLIER,
            "maxLogMultiplier": self.MAX_LOG_MULTIPLIER,
            "motorGroupSize": self.motor_group_size,
            "visualGain": self.VISUAL_GAIN, "recurrentSignal": "2 * rate - 1",
        }

    def observe_status(self) -> dict:
        delta = self.matrix.data[self.plastic_indices] - self._base_plastic_weights
        return {
            "neurons": self.neuron_count, "edges": self.edge_count,
            "fullGraph": True, "dataSource": self.data_source, "minimumSynapses": 1,
            "model": "full-connectome artificial rate approximation",
            "parameters": self._model_parameters(),
            "learningRule": "local recurrent policy gradient with average-reward baseline and eligibility",
            "plasticEdges": int(self.plastic_indices.size),
            "changedWeights": int(np.count_nonzero(delta)),
            "weightDeltaNorm": float(np.linalg.norm(delta)),
            "weightMultiplierMin": float(np.exp(self.log_multipliers.min())),
            "weightMultiplierMax": float(np.exp(self.log_multipliers.max())),
            "actionProbabilities": self.action_probabilities.tolist(), "actions": list(ACTIONS),
            "activationMean": float(self.state.mean()), "activationMax": float(self.state.max()),
            "activeNeurons": int(np.count_nonzero(self.state > 0)),
            "neuralSteps": self.neural_steps, "stepsPerAction": self.RECURRENT_STEPS,
            "frames": self.frames, "learningUpdates": self.learning_updates,
            "manualReinforcements": self.manual_reinforcements,
            "manualRewardTotal": self.manual_reward_total, "canReinforce": self._can_reinforce,
            "rewardBaseline": self.baseline, "totalReward": self.total_reward,
            "lastReward": self.last_reward, "lastAdvantage": self.last_advantage,
            "lastAction": self.last_action, "seed": self.seed,
            "graphChecksum": self.graph_checksum, "backend": dict(self.operator.status),
        }

    def _checkpoint_arrays(self) -> dict[str, np.ndarray]:
        metadata = {
            "version": self.VERSION, "graphChecksum": self.graph_checksum,
            "layoutChecksum": self._layout_checksum, "seed": self.seed,
            "parameters": self._model_parameters(),
            "neurons": self.neuron_count, "edges": self.edge_count,
            "frames": self.frames, "neuralSteps": self.neural_steps,
            "learningUpdates": self.learning_updates, "baseline": self.baseline,
            "manualReinforcements": self.manual_reinforcements,
            "manualRewardTotal": self.manual_reward_total, "canReinforce": self._can_reinforce,
            "totalReward": self.total_reward, "lastReward": self.last_reward,
            "lastAdvantage": self.last_advantage, "lastAction": self.last_action,
            "pending": self._pending, "pendingLearn": self._pending_learn,
            "rng": self.rng.bit_generator.state,
        }
        return {
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
            "state": self.state, "log_multipliers": self.log_multipliers,
            "eligibility": self.eligibility, "pending_gradient": self._pending_gradient,
            "action_probabilities": self.action_probabilities,
        }

    def save(self, path: Path) -> None:
        """Atomic, checksummed NPZ; graph rebuilt from validated source on load."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._checkpoint_arrays()
        arrays["checksum"] = np.asarray(_array_digest(arrays))
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path: Path) -> None:
        """Validate the complete checkpoint before mutating this brain.

        Construct with the saved seed and source graph. Backend may differ;
        exact replay requires the same backend's floating-point arithmetic.
        """
        expected = self._checkpoint_arrays()
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                if set(archive.files) != set(expected) | {"checksum"}:
                    raise ValueError("unexpected checkpoint fields")
                arrays = {key: archive[key] for key in expected}
                if archive["checksum"].shape != () or str(archive["checksum"].item()) != _array_digest(arrays):
                    raise ValueError("checkpoint checksum mismatch")
            metadata = json.loads(str(arrays["metadata"].item()))
            if (metadata["version"] != self.VERSION
                    or metadata["graphChecksum"] != self.graph_checksum
                    or metadata["layoutChecksum"] != self._layout_checksum
                    or metadata["parameters"] != self._model_parameters()
                    or metadata["seed"] != self.seed
                    or metadata["neurons"] != self.neuron_count
                    or metadata["edges"] != self.edge_count):
                raise ValueError("checkpoint does not match this graph, model or seed")
            for key in expected.keys() - {"metadata"}:
                value = arrays[key]
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid checkpoint array: {key}")
            if np.any(arrays["state"] < 0) or np.any(arrays["state"] > 1):
                raise ValueError("checkpoint state is outside the rate bounds")
            theta = arrays["log_multipliers"]
            if np.any(theta < self.MIN_LOG_MULTIPLIER - 1e-6) or np.any(theta > self.MAX_LOG_MULTIPLIER + 1e-6):
                raise ValueError("checkpoint weights are outside the multiplier bounds")
            probabilities = arrays["action_probabilities"]
            if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
                raise ValueError("invalid checkpoint action probabilities")
            for key in ("baseline", "totalReward", "lastReward", "lastAdvantage", "manualRewardTotal"):
                if not np.isfinite(metadata[key]):
                    raise ValueError(f"invalid checkpoint value: {key}")
            for key in ("frames", "neuralSteps", "learningUpdates", "manualReinforcements"):
                if type(metadata[key]) is not int or metadata[key] < 0:
                    raise ValueError(f"invalid checkpoint counter: {key}")
            if metadata["lastAction"] not in range(-1, len(ACTIONS)):
                raise ValueError("invalid checkpoint action")
            if type(metadata["pending"]) is not bool or type(metadata["pendingLearn"]) is not bool:
                raise ValueError("invalid checkpoint feedback state")
            if type(metadata["canReinforce"]) is not bool:
                raise ValueError("invalid checkpoint manual reward state")
            rng = np.random.default_rng()
            rng.bit_generator.state = metadata["rng"]
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid checkpoint metadata") from error

        self.state[:] = arrays["state"]
        self.log_multipliers[:] = arrays["log_multipliers"]
        self.eligibility[:] = arrays["eligibility"]
        self._pending_gradient[:] = arrays["pending_gradient"]
        self.action_probabilities[:] = arrays["action_probabilities"]
        values = self._base_plastic_weights * np.exp(self.log_multipliers)
        self.matrix.data[self.plastic_indices] = values
        self.operator.update_weights(self.plastic_indices, values)
        self.rng = rng
        self.frames = metadata["frames"]
        self.neural_steps = metadata["neuralSteps"]
        self.learning_updates = metadata["learningUpdates"]
        self.manual_reinforcements = metadata["manualReinforcements"]
        self.manual_reward_total = metadata["manualRewardTotal"]
        self._can_reinforce = metadata["canReinforce"]
        self.baseline = metadata["baseline"]
        self.total_reward = metadata["totalReward"]
        self.last_reward = metadata["lastReward"]
        self.last_advantage = metadata["lastAdvantage"]
        self.last_action = metadata["lastAction"]
        self._pending = metadata["pending"]
        self._pending_learn = metadata["pendingLearn"]
