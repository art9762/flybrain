"""Preserve a paused local session before a code restart, then serve it again.

The older room snapshot did not export the procedural reel index. Legacy
captures preserve brain, pose, clocks, rewards, bankroll and station usage;
their reel index restarts at zero. No trained weights are reset.
"""

import argparse
import base64
from collections import deque
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def capture(directory):
    directory.mkdir(parents=True, exist_ok=False)
    latest = ROOT / "outputs" / "brain-latest.npz"
    if latest.exists():
        shutil.copy2(latest, directory / "previous-manual-save.npz")

    def request(path, payload):
        req = urllib.request.Request("http://127.0.0.1:8765" + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    status = request("/api/control", {"action": "pause"})
    request("/api/checkpoint", {"action": "save"})
    shutil.copy2(latest, directory / "brain.npz")
    (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"saved": str(directory), "steps": status["run"]["steps"],
                      "bank": status["world"]["bank"]}))


def restore(directory):
    from flybrain.brain import Brain
    from flybrain.session import LabSession
    from flybrain.world import World

    state = json.loads((directory / "status.json").read_text())
    run = state["run"]
    session = LabSession(ROOT)
    session.world = World.from_snapshot(state["world"])
    session.backend = run["backend_requested"]
    session.brain = Brain(ROOT / "data", seed=42, backend=session.backend)
    session.brain.load(directory / "brain.npz")
    session.learning = run["learning"]
    session.speed = run["speed"]
    session.tick_count = run["steps"]
    session.last_action = run["last_action"]
    session.last_effective_action = run.get("last_effective_action", run["last_action"])
    session.last_step_ms = run["step_ms"]
    session.manual_total = run["manual_reward"]
    session.reward_history = deque(state["rewards"], maxlen=240)
    session.manual_events = deque(state["manual_events"], maxlen=30)
    eye = state["eye"]
    session.last_frame = np.frombuffer(base64.b64decode(eye["rgb"]), dtype=np.uint8).reshape(
        eye["height"], eye["width"], 3).copy()
    return session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "serve"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.directory)
    else:
        import uvicorn
        from flybrain import lab

        lab.session = restore(args.directory)
        print("Restored trained brain and room; paused on " + lab.session.backend, flush=True)
        uvicorn.run(lab.app, host="127.0.0.1", port=8765, access_log=False)
