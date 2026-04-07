# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the GridMind Environment.

Rich observation model with both ASCII grid and structured JSON output,
enabling LLM-native agents to reason over the grid state.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PathosAction(Action):
    """Action for the GridMind environment.

    The ``message`` field accepts either:
    - A digit string: "0"=up, "1"=down, "2"=left, "3"=right
    - A natural language string (parsed by the server): "go north", "move right", etc.
    """

    message: str = Field(..., description="Action as digit (0-3) or natural language direction")


class PathosObservation(Observation):
    """Rich observation from the GridMind environment.

    Provides both a human-readable ASCII grid and a machine-readable
    structured dict for LLM agents to reason over.
    """

    # ── ASCII / human-readable ──────────────────────────────────────────────
    grid_state: str = Field(
        default="",
        description="ASCII grid visualization of the current environment state",
    )

    # ── Backwards-compat alias (kept so existing clients don't break) ────────
    echoed_message: str = Field(
        default="",
        description="[Deprecated] Alias for grid_state. Use grid_state instead.",
    )

    # ── Scalar metrics ──────────────────────────────────────────────────────
    message_length: int = Field(
        default=0,
        description="Current step count (kept for OpenEnv compatibility)",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken this episode",
    )

    # ── Structured observation for LLM agents ──────────────────────────────
    structured: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Machine-readable grid state: agent/goal/trap positions, "
            "Manhattan distance, valid actions, direction hints, etc."
        ),
    )

    # ── Episode state ───────────────────────────────────────────────────────
    keys_collected: int = Field(
        default=0,
        description="Number of keys collected this episode",
    )
    map_type: str = Field(
        default="open",
        description="Current map type: open | sparse | maze | adversarial",
    )
    grid_size: int = Field(
        default=5,
        description="Current grid dimensions (grid_size × grid_size)",
    )
    episode_seed: Optional[int] = Field(
        default=None,
        description="RNG seed used for this episode (None = random)",
    )
