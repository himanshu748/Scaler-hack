"""WebSocket client for the API Design environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ApiDesignAction, ApiDesignObservation, ApiDesignState


class ApiDesignEnv(EnvClient[ApiDesignAction, ApiDesignObservation, ApiDesignState]):
    """Client for interacting with a running ApiDesignEnvironment server."""

    def _step_payload(self, action: ApiDesignAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ApiDesignObservation]:
        obs_data = payload.get("observation", payload)
        obs = ApiDesignObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            requirements=obs_data.get("requirements", ""),
            constraints=obs_data.get("constraints", []),
            feedback=obs_data.get("feedback"),
            suggestions=obs_data.get("suggestions", []),
            attempt_number=obs_data.get("attempt_number", 0),
            max_attempts=obs_data.get("max_attempts", 5),
            total_score=obs_data.get("total_score"),
        )
        return StepResult(
            observation=obs,
            reward=obs_data.get("reward"),
            done=obs_data.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ApiDesignState:
        return ApiDesignState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            problem_id=payload.get("problem_id", ""),
            difficulty=payload.get("difficulty", ""),
            best_score=payload.get("best_score", 0.0),
            max_attempts=payload.get("max_attempts", 5),
        )
