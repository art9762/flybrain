"""Optional training evidence must never prevent a valid model from loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def idle_training_status():
    return {"state": "idle", "steps": 0, "target_steps": 2000, "episodes": 0}


def policy_checksum(brain):
    """Identify the evaluated policy without transient activity or RNG state."""
    digest = hashlib.sha256()
    identity = {"graph": brain.graph_checksum, "seed": brain.seed,
                "version": brain.VERSION, "parameters": brain._model_parameters()}
    digest.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    digest.update(brain.log_multipliers.tobytes())
    return digest.hexdigest()


def load_training_status(path: Path, brain):
    """Recover matching evidence, or show an unscored but playable model.

    Historical v1 reports have no policy checksum; retain compatibility by
    checking their graph and cumulative update count. New reports additionally
    identify the actual learned weights, including after a manual save.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        # Python's JSON reader accepts NaN and overflowing exponents. Such
        # values would make FastAPI's otherwise healthy status response fail.
        json.dumps(report, allow_nan=False)
    except (OSError, ValueError):
        return idle_training_status()
    if not isinstance(report, dict) or report.get("schema") != "flybrain-roulette-training-v1":
        return idle_training_status()
    expected_graph = {"neurons": brain.neuron_count, "edges": brain.edge_count,
                      "checksum": brain.graph_checksum, "plastic_edges": len(brain.plastic_indices)}
    if report.get("graph") != expected_graph:
        return idle_training_status()
    updates = report.get("cumulative_learning_updates")
    if type(updates) is not int or updates != brain.learning_updates:
        return idle_training_status()
    if "policy_checksum" in report and report["policy_checksum"] != policy_checksum(brain):
        return idle_training_status()
    for key in ("steps", "target_steps", "episodes"):
        if type(report.get(key)) is not int or report[key] < 0:
            return idle_training_status()
    return {**report, "state": "stopped" if report.get("stop_reason") == "user_stop" else "complete"}


def save_training_report(path: Path, report):
    """Replace the companion JSON only after serialization and writing finish."""
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
