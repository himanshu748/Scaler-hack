from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


class EndpointSpec(Action):
    """A single API endpoint specification submitted by the agent."""

    method: str = Field(description="HTTP method: GET, POST, PUT, PATCH, DELETE")
    path: str = Field(description="URL path, e.g. /users/{id}/posts")
    description: str = Field(default="", description="What this endpoint does")
    request_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request body schema as field_name -> type_hint",
    )
    response_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response body schema as field_name -> type_hint",
    )
    status_code: int = Field(default=200, description="Expected success status code")
    query_params: List[str] = Field(
        default_factory=list, description="Supported query parameters"
    )


class ApiDesignAction(Action):
    """The agent's submitted API design: a list of endpoint specifications."""

    endpoints: List[EndpointSpec] = Field(
        description="List of endpoint specifications"
    )


class ScoreFeedback(Observation):
    """Per-dimension scoring breakdown."""

    completeness: float = Field(default=0.0, description="Coverage of requirements")
    restful_conventions: float = Field(default=0.0, description="REST best practices")
    schema_quality: float = Field(default=0.0, description="Request/response quality")
    consistency: float = Field(default=0.0, description="Naming and format uniformity")
    http_semantics: float = Field(default=0.0, description="Method safety/idempotency")


class ApiDesignObservation(Observation):
    """Observation returned after each step."""

    requirements: str = Field(default="", description="The problem requirements text")
    constraints: List[str] = Field(
        default_factory=list, description="Specific constraints to satisfy"
    )
    feedback: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension score breakdown"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement hints for the agent"
    )
    attempt_number: int = Field(default=0, description="Current attempt (1-indexed)")
    max_attempts: int = Field(default=5, description="Maximum allowed attempts")
    total_score: Optional[float] = Field(
        default=None, description="Weighted total score 0.0-1.0"
    )


class ApiDesignState(State):
    """Episode state metadata."""

    problem_id: str = Field(default="", description="Current problem identifier")
    difficulty: str = Field(default="", description="easy / medium / hard")
    best_score: float = Field(
        default=0.0, description="Best total score achieved so far"
    )
    max_attempts: int = Field(default=5)
