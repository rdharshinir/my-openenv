# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import PathosAction, PathosObservation
except ImportError:
    from models import PathosAction, PathosObservation


class PathosEnv(
    EnvClient[PathosAction, PathosObservation, State]
):
    """
    Client for the Pathos AI Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with PathosEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(PathosAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PathosEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PathosAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PathosAction) -> Dict:
        """
        Convert PathosAction to JSON payload for step message.

        Args:
            action: PathosAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PathosObservation]:
        """
        Parse server response into StepResult[PathosObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PathosObservation
        """
        obs_data = payload.get("observation", {})
        observation = PathosObservation(
            grid_state=obs_data.get("grid_state", ""),
            echoed_message=obs_data.get("echoed_message", ""),
            structured=obs_data.get("structured", {}),
            message_length=obs_data.get("message_length", 0),
            step_count=obs_data.get("step_count", 0),
            map_type=obs_data.get("map_type", "open"),
            grid_size=obs_data.get("grid_size", 5),
            episode_seed=obs_data.get("episode_seed"),
            keys_collected=obs_data.get("keys_collected", 0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
