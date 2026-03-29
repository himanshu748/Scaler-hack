"""API Design RL Environment -- core logic."""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from ..models import ApiDesignAction, ApiDesignObservation, ApiDesignState
from .grader import grade
from .problems import PROBLEMS, Problem


class ApiDesignEnvironment(Environment):
    """
    RL environment where an agent designs REST API endpoints.

    Each episode presents a problem (set of functional requirements).
    The agent submits endpoint designs and receives multi-dimensional
    feedback with partial credit at every step.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 5

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._state = ApiDesignState()
        self._problem: Optional[Problem] = None
        self._attempt = 0
        self._best_score = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ApiDesignObservation:
        if seed is not None:
            random.seed(seed)

        self._problem = random.choice(PROBLEMS)
        self._attempt = 0
        self._best_score = 0.0

        self._state = ApiDesignState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            problem_id=self._problem["id"],
            difficulty=self._problem["difficulty"],
            best_score=0.0,
            max_attempts=self.MAX_ATTEMPTS,
        )

        return ApiDesignObservation(
            done=False,
            reward=None,
            requirements=self._problem["description"],
            constraints=self._problem["constraints"],
            feedback=None,
            suggestions=[
                f"Design endpoints for: {self._problem['title']}",
                f"Difficulty: {self._problem['difficulty']}",
                f"You have {self.MAX_ATTEMPTS} attempts.",
            ],
            attempt_number=0,
            max_attempts=self.MAX_ATTEMPTS,
            total_score=None,
        )

    def step(
        self,
        action: ApiDesignAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ApiDesignObservation:
        if self._problem is None:
            return ApiDesignObservation(
                done=True,
                reward=0.0,
                requirements="",
                constraints=[],
                feedback=None,
                suggestions=["Call reset() before step()."],
                attempt_number=0,
                max_attempts=self.MAX_ATTEMPTS,
                total_score=0.0,
            )

        self._attempt += 1
        self._state.step_count += 1

        submitted = []
        for ep in action.endpoints:
            submitted.append(ep.model_dump(exclude={"metadata"}))

        result = grade(submitted, self._problem["ground_truth"])

        total = result["total"]
        self._best_score = max(self._best_score, total)
        self._state.best_score = self._best_score

        perfect = total >= 0.95
        exhausted = self._attempt >= self.MAX_ATTEMPTS
        done = perfect or exhausted

        if perfect:
            result["suggestions"] = ["Excellent API design! Near-perfect score."]
        elif done:
            result["suggestions"].insert(
                0, f"Final attempt reached. Best score: {self._best_score:.2f}"
            )

        return ApiDesignObservation(
            done=done,
            reward=total,
            requirements=self._problem["description"],
            constraints=self._problem["constraints"],
            feedback=result["scores"],
            suggestions=result["suggestions"],
            attempt_number=self._attempt,
            max_attempts=self.MAX_ATTEMPTS,
            total_score=total,
        )

    @property
    def state(self) -> ApiDesignState:
        return self._state
