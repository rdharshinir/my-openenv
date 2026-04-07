# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the GridMind Environment.

Exposes endpoints for the RL environment, and adds extra endpoints 
for the Hackathon UI like Leaderboard and Static files.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import sqlite3
import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PathosAction, PathosObservation
    from .my_env_environment import PathosEnvironment
except ImportError:
    from models import PathosAction, PathosObservation
    from server.my_env_environment import PathosEnvironment


# Create the app with web interface and README integration
app = create_app(
    PathosEnvironment,
    PathosAction,
    PathosObservation,
    env_name="gridmind",
    max_concurrent_envs=10,  # Increased for multi-agent or concurrent demo usage
)


# ── Hackathon Additions (Leaderboard) ──────────────────────────────────

LEADERBOARD_FILE = "leaderboard.sqlite"

def init_db():
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            score REAL,
            episodes INT,
            avg_steps REAL
        )
    ''')
    # Add mockup data if empty
    c.execute("SELECT count(*) FROM leaderboard")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO leaderboard (agent_name, score, episodes, avg_steps) VALUES ('Llama-4-Scout-17B', 0.95, 100, 15.2)")
        c.execute("INSERT INTO leaderboard (agent_name, score, episodes, avg_steps) VALUES ('GPT-4o', 0.92, 100, 16.1)")
        c.execute("INSERT INTO leaderboard (agent_name, score, episodes, avg_steps) VALUES ('Rule-Based BFS', 0.88, 100, 14.0)")
    conn.commit()
    conn.close()

init_db()

@app.get("/leaderboard")
async def get_leaderboard():
    """Retrieve leaderboard scores."""
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute("SELECT agent_name, score, episodes, avg_steps FROM leaderboard ORDER BY score DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    
    results = [
        {"rank": i + 1, "agent_name": r[0], "score": r[1], "episodes": r[2], "avg_steps": r[3]}
        for i, r in enumerate(rows)
    ]
    return JSONResponse(content={"leaderboard": results})

@app.post("/submit_score")
async def submit_score(request: Request):
    """Submit a new score to the leaderboard."""
    data = await request.json()
    conn = sqlite3.connect(LEADERBOARD_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO leaderboard (agent_name, score, episodes, avg_steps) VALUES (?, ?, ?, ?)",
              (data.get("agent_name"), data.get("score"), data.get("episodes"), data.get("avg_steps")))
    conn.commit()
    conn.close()
    return JSONResponse(content={"status": "success"})


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
