# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GridMind – Grid-World Environment Implementation.

A grid-world RL environment exposed over HTTP/WebSocket via OpenEnv.

Features
--------
- Randomized goal position every episode
- Increasing grid size based on episode count
- Multiple map types (maze, sparse, adversarial)
- Rich structured observation for LLM agents
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PathosAction, PathosObservation
    from ..env import GridEnv
except ImportError:
    from models import PathosAction, PathosObservation
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env import GridEnv


class PathosEnvironment(Environment):
    """
    Grid-world environment with randomized goals, increasing difficulty,
    step-penalty tuning, and structure observations.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_count = 0
        self._grid = GridEnv(episode=0)

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self, seed: int = None) -> PathosObservation:
        """Reset the grid, advance difficulty, return rich observation."""
        self._episode_count += 1
        
        # Advance difficulty and optionally use a specific seed
        self._grid.reset(advance_difficulty=True, seed=seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)

        rendered_grid = self._grid.render()
        structured_obs = self._grid.structured_obs()

        return PathosObservation(
            grid_state=rendered_grid,
            echoed_message=rendered_grid, # Backwards compat
            structured=structured_obs,
            message_length=0,
            step_count=0,
            map_type=self._grid.map_type,
            grid_size=self._grid.size,
            episode_seed=self._grid.seed,
            keys_collected=self._grid.keys_collected,
            done=False,
            reward=0.0,
            metadata=self._meta(result="reset", structured=structured_obs),
        )

    def step(self, action: PathosAction) -> PathosObservation:  # type: ignore[override]
        """
        Execute one grid step.
        Supports natural language parsing for basic directions or digits.
        """
        self._state.step_count += 1

        msg = action.message.strip().lower()

        # Simple naive natural language to action parsing
        act_int = -1
        if msg in ("0", "up", "north"): act_int = 0
        elif msg in ("1", "down", "south"): act_int = 1
        elif msg in ("2", "left", "west"): act_int = 2
        elif msg in ("3", "right", "east"): act_int = 3
        else:
            # Fallback parsing just in case it contains keywords
            if "up" in msg or "north" in msg: act_int = 0
            elif "down" in msg or "south" in msg: act_int = 1
            elif "left" in msg or "west" in msg: act_int = 2
            elif "right" in msg or "east" in msg: act_int = 3

        if act_int == -1:
            reward, done, info = GridEnv.STEP_PENALTY, False, {"result": "invalid_action"}
        else:
            _, reward, done, info = self._grid.step(act_int)

        rendered_grid = self._grid.render()
        structured_obs = self._grid.structured_obs()

        return PathosObservation(
            grid_state=rendered_grid,
            echoed_message=rendered_grid,
            structured=structured_obs,
            message_length=self._grid.steps,
            step_count=self._grid.steps,
            map_type=self._grid.map_type,
            grid_size=self._grid.size,
            episode_seed=self._grid.seed,
            keys_collected=self._grid.keys_collected,
            done=done,
            reward=float(reward),
            metadata=self._meta(structured=structured_obs, **info),
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Helpers ────────────────────────────────────────────────────────────

    def _meta(self, **extra) -> dict:
        return {
            "grid_size":    self._grid.size,
            "agent":        self._grid.agent,
            "goal":         self._grid.goal,
            "traps":        self._grid.traps,
            "step":         self._grid.steps,
            "episode":      self._episode_count,
            **extra,
        }
