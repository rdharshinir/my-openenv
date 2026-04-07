"""
Pathos AI – GridMind Environment Implementation (OpenEnv server wrapper).
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
    Grid-world environment with:
    - 4 curriculum difficulty levels
    - Wind zones, fog of war, dynamic hazards
    - Tiered objectives (extraction, survivors, speed bonus)
    - Replay trajectory recording
    - Scenario editor support (custom layouts)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_count = 0
        self._grid = GridEnv(episode=0)

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: int = None,
        difficulty: int = None,
        custom_layout: dict = None,
    ) -> PathosObservation:
        """Reset with optional curriculum difficulty or custom scenario layout."""
        self._episode_count += 1
        self._grid.reset(
            advance_difficulty=(difficulty is None and custom_layout is None),
            seed=seed,
            difficulty=difficulty,
            custom_layout=custom_layout,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

        rendered = self._grid.render()
        structured = self._grid.structured_obs()
        grid_ui = self._grid.get_grid_for_ui()

        return PathosObservation(
            grid_state=rendered,
            echoed_message=rendered,
            structured=structured,
            grid_ui=grid_ui,
            message_length=0,
            step_count=0,
            reward=0.0,
            done=False,
            map_type=self._grid.map_type,
            grid_size=self._grid.size,
            episode_seed=self._grid.seed,
            difficulty_level=self._grid.difficulty_level,
            difficulty_label=self._grid.difficulty_label,
            keys_collected=self._grid.keys_collected,
            objectives=dict(self._grid.objectives),
            trajectory=[],
            metadata=self._meta(result="reset", structured=structured),
        )

    def step(self, action: PathosAction) -> PathosObservation:
        """Execute one grid step with natural language or digit action."""
        self._state.step_count += 1
        msg = action.message.strip().lower()

        act_int = -1
        if   msg in ("0", "up",    "north"): act_int = 0
        elif msg in ("1", "down",  "south"): act_int = 1
        elif msg in ("2", "left",  "west"):  act_int = 2
        elif msg in ("3", "right", "east"):  act_int = 3
        else:
            if   "up"    in msg or "north" in msg: act_int = 0
            elif "down"  in msg or "south" in msg: act_int = 1
            elif "left"  in msg or "west"  in msg: act_int = 2
            elif "right" in msg or "east"  in msg: act_int = 3

        if act_int == -1:
            reward, done, info = GridEnv.STEP_PENALTY, False, {"result": "invalid_action"}
        else:
            _, reward, done, info = self._grid.step(act_int)

        rendered = self._grid.render()
        structured = self._grid.structured_obs()
        grid_ui = self._grid.get_grid_for_ui()

        return PathosObservation(
            grid_state=rendered,
            echoed_message=rendered,
            structured=structured,
            grid_ui=grid_ui,
            message_length=self._grid.steps,
            step_count=self._grid.steps,
            reward=float(reward),
            done=done,
            map_type=self._grid.map_type,
            grid_size=self._grid.size,
            episode_seed=self._grid.seed,
            difficulty_level=self._grid.difficulty_level,
            difficulty_label=self._grid.difficulty_label,
            keys_collected=self._grid.keys_collected,
            objectives=dict(self._grid.objectives),
            trajectory=list(self._grid.trajectory),
            metadata=self._meta(structured=structured, **info),
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Helpers ────────────────────────────────────────────────────────────

    def _meta(self, **extra) -> dict:
        return {
            "grid_size": self._grid.size,
            "agent":     self._grid.agent,
            "goal":      self._grid.goal,
            "traps":     self._grid.traps,
            "step":      self._grid.steps,
            "episode":   self._episode_count,
            **extra,
        }

    # ── Extra methods exposed to app.py ───────────────────────────────────

    def get_heatmap(self) -> dict:
        return {
            "heatmap": dict(self._grid.visit_heatmap),
            "size": self._grid.size,
        }

    def get_replay(self) -> dict:
        return {
            "current":  self._grid.trajectory,
            "best":     self._grid.best_trajectory,
            "worst":    self._grid.worst_trajectory,
            "best_score":  self._grid._best_score if self._grid.best_trajectory else None,
            "worst_score": self._grid._worst_score if self._grid.worst_trajectory else None,
        }

    def get_layout(self) -> dict:
        return self._grid.export_layout()

    def load_layout(self, layout: dict):
        self._grid.reset(advance_difficulty=False, custom_layout=layout)
