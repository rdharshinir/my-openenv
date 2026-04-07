"""
FastAPI application for the Pathos AI Drone Rescue Simulator.

Extra endpoints beyond OpenEnv standard:
  GET  /ui                  → Serve the visual dashboard
  GET  /grid_ui             → Live SVG cell data for the renderer
  GET  /heatmap             → Visit frequency heatmap
  GET  /replay              → Current + best + worst trajectories
  POST /load_layout         → Load a scenario editor JSON seed
  GET  /export_layout       → Export current map as JSON seed
  GET  /leaderboard         → Top-10 scores
  POST /submit_score        → Submit a new score
  GET  /episode_stats       → Live reward curve data
"""

import json
import os
import sqlite3
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'uv sync'"
    ) from e

try:
    from ..models import PathosAction, PathosObservation
    from .my_env_environment import PathosEnvironment
except ImportError:
    from models import PathosAction, PathosObservation
    from server.my_env_environment import PathosEnvironment


# ── Create OpenEnv base app ─────────────────────────────────────────────
app = create_app(
    PathosEnvironment,
    PathosAction,
    PathosObservation,
    env_name="gridmind",
    max_concurrent_envs=10,
)

# ── Static files (dashboard HTML) ───────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Leaderboard (SQLite) ─────────────────────────────────────────────────
LEADERBOARD_FILE = "leaderboard.sqlite"
EPISODE_LOG_FILE = "episode_log.json"


def init_db():
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            score      REAL,
            episodes   INT,
            avg_steps  REAL,
            success    INTEGER DEFAULT 0,
            difficulty TEXT DEFAULT 'Custom',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("SELECT count(*) FROM leaderboard")
    if c.fetchone()[0] == 0:
        seed_data = [
            ("Llama-4-Scout-17B", 0.95, 100, 15.2, 1, "Legendary"),
            ("GPT-4o",            0.92, 100, 16.1, 1, "Expert"),
            ("Rule-Based BFS",    0.88, 100, 14.0, 1, "Trained"),
            ("Random Agent",      0.21,  50, 32.5, 0, "Rookie"),
        ]
        c.executemany(
            "INSERT INTO leaderboard (agent_name,score,episodes,avg_steps,success,difficulty) VALUES (?,?,?,?,?,?)",
            seed_data,
        )
    conn.commit()
    conn.close()


def load_episode_log() -> list:
    if os.path.exists(EPISODE_LOG_FILE):
        try:
            with open(EPISODE_LOG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_episode_log(log: list):
    with open(EPISODE_LOG_FILE, "w") as f:
        json.dump(log[-200:], f)  # keep last 200 episodes


init_db()


# ── Dashboard ────────────────────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main visual dashboard."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found. Check server/static/index.html</h1>", status_code=404)


# ── Grid UI endpoint ─────────────────────────────────────────────────────

@app.get("/grid_ui/{episode_id}")
async def get_grid_ui(episode_id: str):
    """Return the live SVG grid cell data for a session."""
    # Access the environment pool managed by OpenEnv
    try:
        env_pool = app.state.env_pool if hasattr(app.state, "env_pool") else None
        if env_pool and episode_id in env_pool:
            env: PathosEnvironment = env_pool[episode_id]
            return JSONResponse(content=env._grid.get_grid_for_ui())
    except Exception:
        pass
    return JSONResponse(content={"error": "session not found"}, status_code=404)


# ── Heatmap ──────────────────────────────────────────────────────────────

@app.get("/heatmap/{episode_id}")
async def get_heatmap(episode_id: str):
    """Return visit frequency heatmap for a session."""
    try:
        env_pool = app.state.env_pool if hasattr(app.state, "env_pool") else None
        if env_pool and episode_id in env_pool:
            env: PathosEnvironment = env_pool[episode_id]
            return JSONResponse(content=env.get_heatmap())
    except Exception:
        pass
    return JSONResponse(content={"error": "session not found"}, status_code=404)


# ── Replay ───────────────────────────────────────────────────────────────

@app.get("/replay/{episode_id}")
async def get_replay(episode_id: str):
    """Return current, best, and worst episode trajectories."""
    try:
        env_pool = app.state.env_pool if hasattr(app.state, "env_pool") else None
        if env_pool and episode_id in env_pool:
            env: PathosEnvironment = env_pool[episode_id]
            return JSONResponse(content=env.get_replay())
    except Exception:
        pass
    return JSONResponse(content={"error": "session not found"}, status_code=404)


# ── Scenario editor ──────────────────────────────────────────────────────

@app.get("/export_layout/{episode_id}")
async def export_layout(episode_id: str):
    """Export current map layout as a JSON seed."""
    try:
        env_pool = app.state.env_pool if hasattr(app.state, "env_pool") else None
        if env_pool and episode_id in env_pool:
            env: PathosEnvironment = env_pool[episode_id]
            return JSONResponse(content=env.get_layout())
    except Exception:
        pass
    return JSONResponse(content={"error": "session not found"}, status_code=404)


@app.post("/load_layout/{episode_id}")
async def load_layout(episode_id: str, request: Request):
    """Load a JSON seed into the current session."""
    data = await request.json()
    try:
        env_pool = app.state.env_pool if hasattr(app.state, "env_pool") else None
        if env_pool and episode_id in env_pool:
            env: PathosEnvironment = env_pool[episode_id]
            env.load_layout(data)
            return JSONResponse(content={"status": "ok", "layout": env.get_layout()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    return JSONResponse(content={"error": "session not found"}, status_code=404)


# ── Episode stats for reward curve ──────────────────────────────────────

@app.get("/episode_stats")
async def episode_stats():
    """Return last 50 episode reward/success data for the reward curve chart."""
    log = load_episode_log()
    last_50 = log[-50:]
    return JSONResponse(content={"episodes": last_50})


@app.post("/log_episode")
async def log_episode(request: Request):
    """Log an episode result (called from the UI after each episode end)."""
    data = await request.json()
    log = load_episode_log()
    log.append({
        "episode": len(log) + 1,
        "total_reward": data.get("total_reward", 0),
        "steps": data.get("steps", 0),
        "success": data.get("success", False),
        "difficulty": data.get("difficulty", "Custom"),
    })
    save_episode_log(log)
    success_count = sum(1 for e in log if e.get("success"))
    return JSONResponse(content={
        "status": "logged",
        "total_episodes": len(log),
        "success_rate": round(success_count / len(log) * 100, 1),
    })


# ── Leaderboard ──────────────────────────────────────────────────────────

@app.get("/leaderboard")
async def get_leaderboard():
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT agent_name, score, episodes, avg_steps, success, difficulty
        FROM leaderboard ORDER BY score DESC LIMIT 10
    """)
    rows = c.fetchall()
    conn.close()
    results = [
        {
            "rank": i + 1,
            "agent_name": r[0],
            "score": r[1],
            "episodes": r[2],
            "avg_steps": r[3],
            "success": bool(r[4]),
            "difficulty": r[5],
        }
        for i, r in enumerate(rows)
    ]
    return JSONResponse(content={"leaderboard": results})


@app.post("/submit_score")
async def submit_score(request: Request):
    data = await request.json()
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO leaderboard (agent_name, score, episodes, avg_steps, success, difficulty) VALUES (?,?,?,?,?,?)",
        (
            data.get("agent_name", "Anonymous"),
            data.get("score", 0.0),
            data.get("episodes", 1),
            data.get("avg_steps", 0.0),
            1 if data.get("success") else 0,
            data.get("difficulty", "Custom"),
        ),
    )
    conn.commit()
    conn.close()
    return JSONResponse(content={"status": "success"})


# ── Entry point ──────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
