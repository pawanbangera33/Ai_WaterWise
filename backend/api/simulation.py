"""
FastAPI Router for Real-Time Telemetry Data Simulation
Provides controls, condition switching, live single-tick retrieval, history, and Server-Sent Events (SSE) streaming.
"""

import asyncio
import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from backend.services.simulation import simulation_engine

router = APIRouter(prefix="/simulation", tags=["Simulation Engine"])

class SimulationStartRequest(BaseModel):
    condition: Optional[str] = "normal"
    speed: Optional[float] = 1.0

class ConditionChangeRequest(BaseModel):
    condition: str

@router.get("/status")
def get_simulation_status():
    """Returns status of real-time simulation engine."""
    return simulation_engine.get_status()

@router.post("/start")
def start_simulation(req: SimulationStartRequest):
    """Starts real-time simulation streaming."""
    return simulation_engine.start(condition=req.condition, speed=req.speed)

@router.post("/pause")
def pause_simulation():
    """Pauses active real-time simulation."""
    return simulation_engine.pause()

@router.post("/resume")
def resume_simulation():
    """Resumes paused real-time simulation."""
    return simulation_engine.resume()

@router.post("/stop")
def stop_simulation():
    """Stops and resets real-time simulation."""
    return simulation_engine.stop()

@router.post("/condition")
def change_simulation_condition(req: ConditionChangeRequest):
    """Dynamically updates active simulation condition on-the-fly."""
    valid_conditions = ["normal", "major_leak", "micro_leak", "sensor_fault", "high_demand", "low_tank"]
    if req.condition not in valid_conditions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid condition '{req.condition}'. Must be one of {valid_conditions}"
        )
    return simulation_engine.set_condition(req.condition)

@router.get("/tick")
def get_simulation_tick():
    """Advances and returns next telemetry tick evaluated against AI models."""
    if not simulation_engine.is_running:
        # If engine isn't running, start it automatically in normal mode
        simulation_engine.start()
        
    tick_data = simulation_engine.generate_next_tick()
    return {
        "status": simulation_engine.get_status(),
        "tick": tick_data
    }

@router.get("/history")
def get_simulation_history(limit: int = 50):
    """Returns buffered history of simulated ticks."""
    return {
        "count": len(simulation_engine.history),
        "history": simulation_engine.history[-limit:]
    }

@router.get("/stream")
async def stream_simulation_ticks():
    """
    Server-Sent Events (SSE) endpoint pushing live real-time telemetry ticks
    to connected web clients.
    """
    if not simulation_engine.is_running:
        simulation_engine.start()

    async def event_generator():
        while True:
            if simulation_engine.is_running and not simulation_engine.is_paused:
                tick_data = simulation_engine.generate_next_tick()
                payload = {
                    "status": simulation_engine.get_status(),
                    "tick": tick_data
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
                # Wait according to configured speed
                await asyncio.sleep(simulation_engine.speed)
            else:
                # Heartbeat when paused or stopped
                yield f"data: {json.dumps({'status': simulation_engine.get_status(), 'tick': None})}\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
