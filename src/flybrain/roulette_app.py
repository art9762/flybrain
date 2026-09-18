"""Local browser duel against the full FlyWire connectome."""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .roulette_session import RouletteSession

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="FlyBrain Roulette", version="0.1.0")
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
session = RouletteSession(ROOT)

class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Load(Request):
    backend: Literal["auto", "cpu", "mps"] = "auto"

class New(Request):
    starting_player: Literal[0, 1] = 0

class Action(Request):
    action: int = Field(ge=0, le=7, strict=True)

class Training(Request):
    steps: int = Field(default=2000, ge=100, le=20000, strict=True)

class Checkpoint(Request):
    action: Literal["save", "load"]

def call(function, *args):
    try:
        return function(*args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "roulette.html")

@app.get("/api/status")
def status():
    return session.status()

@app.post("/api/load")
def load(body: Load):
    return call(session.load_brain, body.backend)

@app.post("/api/new")
def new(body: New):
    return call(session.new_game, body.starting_player)

@app.post("/api/action")
def action(body: Action):
    return call(session.action, body.action)

@app.post("/api/fly")
def fly():
    return call(session.fly)

@app.post("/api/train")
def training(body: Training):
    return call(session.start_training, body.steps)

@app.post("/api/stop")
def stop():
    return call(session.stop)

@app.post("/api/checkpoint")
def checkpoint(body: Checkpoint):
    return call(session.checkpoint, body.action)
