"""Local-only web server for the embodied learning room."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

from .session import LabSession

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="FlyBrain Room", version="0.2.0")
session = LabSession(ROOT)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    # Raw rejected values can contain Infinity/NaN and are not valid JSON.
    return JSONResponse(status_code=422, content={
        "detail": "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    })


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Control(StrictRequest):
    action: Literal["start", "pause", "step"]
    learning: bool | None = None
    speed: float | None = Field(default=None, ge=0.25, le=64)
    backend: Literal["auto", "cpu", "mps"] | None = None


class Reward(StrictRequest):
    amount: float = Field(ge=-10, le=10, allow_inf_nan=False)


class Credits(StrictRequest):
    amount: int = Field(ge=1, le=10000)


class Checkpoint(StrictRequest):
    action: Literal["save", "load"]


def call(function, *args):
    try:
        return function(*args)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "room.html")


@app.get("/api/status")
def status():
    return session.status()


@app.post("/api/control")
def control(body: Control):
    return call(session.control, body.action, body.learning, body.speed, body.backend)


@app.post("/api/reset-position")
def reset_position():
    return call(session.reset_position)


@app.post("/api/config")
def config(body: dict):
    return call(session.configure, body)


@app.post("/api/reward")
def reward(body: Reward):
    return call(session.reward, body.amount)


@app.post("/api/credits")
def credits(body: Credits):
    return call(session.credits, body.amount)


@app.post("/api/checkpoint")
def checkpoint(body: Checkpoint):
    return call(session.checkpoint, body.action)
