from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .connectome import choose_seed, extract_reachable_subgraph, load_connectome
from .simulator import SimulationConfig, simulate

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="FlyBrain Lab", version="0.1.0")


@lru_cache(maxsize=4)
def get_connectome(minimum_synapses: int):
    return load_connectome(ROOT / "data", minimum_synapses)


class RunRequest(BaseModel):
    seed_index: int | None = Field(default=None, ge=0)
    neurons: int = Field(default=5_000, ge=10, le=20_000)
    hops: int = Field(default=3, ge=1, le=6)
    minimum_synapses: int = Field(default=5, ge=1, le=100)
    duration_ms: float = Field(default=250, ge=10, le=2_000)
    timestep_ms: float = Field(default=0.5, ge=0.1, le=2)
    synaptic_gain: float = Field(default=0.045, ge=0, le=1)
    stimulus_amplitude: float = Field(default=1.25, ge=0, le=10)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/stats")
def stats(minimum_synapses: int = 5) -> dict:
    graph = get_connectome(minimum_synapses)
    seed = choose_seed(graph)
    return {
        "neurons": graph.neuron_count,
        "edges": graph.edge_count,
        "default_seed_index": seed,
        "default_seed_flywire_id": int(graph.neuron_ids[seed]),
        "minimum_synapses": minimum_synapses,
    }


@app.post("/api/simulate")
def run(request: RunRequest) -> dict:
    try:
        started = time.perf_counter()
        full_graph = get_connectome(request.minimum_synapses)
        seed = request.seed_index if request.seed_index is not None else choose_seed(full_graph)
        subgraph = extract_reachable_subgraph(full_graph, seed, request.neurons, request.hops)
        config = SimulationConfig(
            duration_ms=request.duration_ms,
            timestep_ms=request.timestep_ms,
            synaptic_gain=request.synaptic_gain,
            stimulus_amplitude=request.stimulus_amplitude,
            stimulus_start_ms=request.duration_ms * 0.1,
            stimulus_end_ms=request.duration_ms * 0.4,
        )
        result = simulate(subgraph, 0, config).to_dict()
        result["source_seed_index"] = seed
        result["wall_time_seconds"] = round(time.perf_counter() - started, 4)
        return result
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
