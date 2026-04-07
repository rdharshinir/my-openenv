from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PathosAction(Action):
    """Action for the Pathos AI Drone environment.

    message accepts:
    - Digit: "0"=up, "1"=down, "2"=left, "3"=right
    - Natural language: "go north", "move right", etc.
    """
    message: str = Field(..., description="Action as digit (0-3) or natural language direction")


class PathosObservation(Observation):
    """Rich observation from the Pathos AI environment."""

    # ── Rendered grid ──────────────────────────────────────────────────────
    grid_state: str = Field(default="", description="Emoji grid visualization")
    echoed_message: str = Field(default="", description="[Deprecated] Alias for grid_state")

    # ── Step metrics ───────────────────────────────────────────────────────
    message_length: int = Field(default=0, description="Current step count (OpenEnv compat)")
    step_count: int = Field(default=0, description="Steps taken this episode")
    reward: float = Field(default=0.0, description="Reward from last action")
    done: bool = Field(default=False, description="Whether episode has ended")

    # ── Structured machine-readable obs ───────────────────────────────────
    structured: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full machine-readable state for LLM agents",
    )

    # ── Grid state for UI renderer ─────────────────────────────────────────
    grid_ui: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cell-by-cell grid data for SVG/Canvas renderer",
    )

    # ── Episode metadata ───────────────────────────────────────────────────
    map_type: str = Field(default="open", description="Map type: open|sparse|maze|adversarial")
    grid_size: int = Field(default=5, description="Grid dimensions (N x N)")
    episode_seed: Optional[int] = Field(default=None, description="RNG seed for this episode")
    difficulty_level: Optional[int] = Field(default=None, description="Curriculum level 1-4")
    difficulty_label: str = Field(default="Custom", description="Human-readable difficulty name")

    # ── Objectives ─────────────────────────────────────────────────────────
    keys_collected: int = Field(default=0, description="Survivors rescued this episode")
    objectives: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tiered objective statuses: extraction, survivors, speed bonus",
    )

    # ── Replay ─────────────────────────────────────────────────────────────
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full trajectory so far: [{pos, action, reward, done, step}]",
    )

    # ── Metadata ───────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = Field(default_factory=dict)
