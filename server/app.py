from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from indiashield.env import IndiaShieldEnv, Action
from indiashield.tasks import get_all_tasks, get_task
import time
import uuid

app = FastAPI(
    title="IndiaShield-v1",
    description=(
        "A reinforcement learning environment simulating "
        "WhatsApp misinformation containment in India + "
        "MuRIL multilingual model compression for low-RAM "
        "Indian devices. Built for OpenEnv hackathon."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, IndiaShieldEnv] = {}
session_metadata: Dict[str, Dict] = {}
MAX_SESSIONS = 100


def _cleanup_old_sessions():
    if len(sessions) < MAX_SESSIONS:
        return
    now = time.time()
    expired = [
        sid for sid, meta in session_metadata.items()
        if now - meta.get("last_used", 0) > 3600
    ]
    for sid in expired:
        sessions.pop(sid, None)
        session_metadata.pop(sid, None)


def _get_session(session_id: str) -> IndiaShieldEnv:
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Call POST /reset to start a new session."
            )
        )
    session_metadata[session_id]["last_used"] = time.time()
    return env


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = "default"


@app.get("/demo", include_in_schema=False)
def demo():
    return FileResponse("demo.html")


@app.get("/", summary="Environment info", tags=["Info"])
def root():
    return {
        "name": "IndiaShield-v1",
        "version": "1.0.0",
        "tagline": (
            "Contain WhatsApp misinformation in India while "
            "compressing a multilingual AI classifier for "
            "low-RAM Indian devices"
        ),
        "tasks": [
            {
                "id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "users": t.total_users,
                "max_turns": t.max_turns
            }
            for t in get_all_tasks()
        ],
        "actions": [
            "intercept", "quarantine", "identify_spreader",
            "add_forward_label", "quantize", "prune",
            "distill", "deploy", "noop"
        ],
        "active_sessions": len(sessions),
        "docs": "/docs"
    }


@app.get("/health", summary="Health check", tags=["Info"])
def health():
    return {
        "status": "ok",
        "environment": "IndiaShield-v1",
        "active_sessions": len(sessions),
        "timestamp": time.time()
    }


@app.get("/tasks", summary="List all tasks", tags=["Tasks"])
def list_tasks():
    return {
        "tasks": [
            {
                "id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "story": t.story,
                "total_users": t.total_users,
                "max_turns": t.max_turns,
                "initial_model_size_mb": t.initial_model_size_mb,
                "target_model_size_mb": t.target_model_size_mb,
                "language_mix": t.language_mix,
                "win_condition": t.win_condition.model_dump()
            }
            for t in get_all_tasks()
        ]
    }


@app.get("/tasks/{task_id}", summary="Get one task details", tags=["Tasks"])
def get_task_detail(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Valid tasks: task1, task2, task3, task4, task5"
        )
    return task.model_dump()


@app.post("/reset", summary="Start a new episode", tags=["Environment"])
async def reset(request: Request):
    _cleanup_old_sessions()

    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "task1")
    session_id = body.get("session_id", None) or str(uuid.uuid4())
    seed = body.get("seed", None)
    custom_config = body.get("custom_config", {})

    task = get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: task1, task2, task3, task4, task5"
        )

    env = IndiaShieldEnv(
        task_id=task_id,
        seed=seed,
        custom_config=custom_config
    )
    sessions[session_id] = env
    session_metadata[session_id] = {
        "created_at": time.time(),
        "last_used": time.time(),
        "task_id": task_id,
        "seed": env.seed
    }

    obs = env.reset()
    return {
        "session_id": session_id,
        "episode_id": obs.episode_id,
        "task_id": task_id,
        "task_name": obs.task_name,
        "seed": env.seed,
        "observation": obs.model_dump(),
        "message": (
            f"Episode started. Task: {obs.task_name}. "
            f"Seed: {env.seed}. "
            f"Use session_id '{session_id}' for all subsequent calls."
        )
    }


@app.post("/step", summary="Take one action", tags=["Environment"])
def step(request: StepRequest):
    env = _get_session(request.session_id)

    try:
        action = Action(**request.action)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {str(e)}."
        )

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state", summary="Get current state", tags=["Environment"])
def state(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    return env.state()


@app.get("/grade", summary="Get current score", tags=["Environment"])
def grade(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    result = env.grade()
    return result.model_dump()


@app.get("/stats", summary="Get episode statistics", tags=["Environment"])
def stats(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    return env.get_stats().model_dump()


@app.delete("/session/{session_id}", summary="Delete a session", tags=["Environment"])
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    sessions.pop(session_id)
    session_metadata.pop(session_id, None)
    return {"message": f"Session '{session_id}' deleted"}


@app.get("/sessions", summary="List active sessions", tags=["Info"])
def list_sessions():
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "task_id": meta.get("task_id"),
                "seed": meta.get("seed"),
                "created_at": meta.get("created_at"),
                "last_used": meta.get("last_used"),
                "turn": sessions[sid].turn,
                "done": sessions[sid].done
            }
            for sid, meta in session_metadata.items()
        ]
    }


@app.get("/actions", summary="List all valid actions", tags=["Info"])
def list_actions():
    return {
        "actions": [
            {"type": "intercept", "description": "Block one infected node", "example": {"type": "intercept", "node_id": 42}},
            {"type": "quarantine", "description": "Block entire group", "example": {"type": "quarantine", "group_id": 3}},
            {"type": "identify_spreader", "description": "Find highest-risk node", "example": {"type": "identify_spreader"}},
            {"type": "add_forward_label", "description": "Mark message as forwarded", "example": {"type": "add_forward_label", "node_id": 7}},
            {"type": "quantize", "description": "Shrink model by reducing precision", "example": {"type": "quantize", "precision": "int8"}},
            {"type": "prune", "description": "Cut unused model weights", "example": {"type": "prune", "target_layer": "ffn", "percentage": 30}},
            {"type": "distill", "description": "Rebuild into tiny student model", "example": {"type": "distill", "student_size": "small"}},
            {"type": "deploy", "description": "Install model on phone", "example": {"type": "deploy", "node_id": 15}},
            {"type": "noop", "description": "Do nothing — penalised", "example": {"type": "noop"}}
        ]
    }


@app.get("/network/nodes", summary="Get all nodes", tags=["Environment"])
def get_nodes(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    nodes = []
    for node_id, node in env.network.nodes.items():
        nodes.append({
            "id": node.id,
            "name": node.name,
            "group_id": node.group_id,
            "group_type": node.group_type,
            "status": node.status,
            "is_super_spreader": node.is_super_spreader,
            "is_source": node.is_source,
            "has_misinfo": node.has_misinfo,
            "connections": node.connections,
            "forward_label_applied": node.forward_label_applied,
            "message_id": node.message_id
        })
    groups = []
    for group_id, group in env.network.groups.items():
        groups.append({
            "id": group.id,
            "name": group.name,
            "group_type": group.group_type,
            "member_count": len(group.member_ids),
            "is_quarantined": group.is_quarantined,
            "trust_multiplier": group.trust_multiplier,
            "spread_multiplier": group.spread_multiplier,
            "skepticism": group.skepticism
        })
    return {
        "session_id": session_id,
        "turn": env.turn,
        "nodes": nodes,
        "groups": groups,
        "summary": env.network.get_state().model_dump()
    }


@app.get("/network/infected", summary="Get infected nodes", tags=["Environment"])
def get_infected_nodes(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    infected = [
        {
            "id": node.id,
            "name": node.name,
            "group_id": node.group_id,
            "group_type": node.group_type,
            "is_super_spreader": node.is_super_spreader,
            "is_source": node.is_source,
            "connections": len(node.connections),
            "message_id": node.message_id
        }
        for node in env.network.nodes.values()
        if node.status == "infected"
    ]
    blocked = [
        {
            "id": node.id,
            "name": node.name,
            "group_id": node.group_id,
            "group_type": node.group_type,
        }
        for node in env.network.nodes.values()
        if node.status == "blocked"
    ]
    return {
        "session_id": session_id,
        "turn": env.turn,
        "infected_nodes": infected,
        "blocked_nodes": blocked,
        "infected_count": len(infected),
        "blocked_count": len(blocked)
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()