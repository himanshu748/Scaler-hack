"""API Design RL Environment -- core logic.

A production-relevant RL environment where an AI agent learns to design
REST APIs.  Given a set of functional requirements the agent submits
endpoint specifications and receives multi-dimensional, partial-credit
feedback at every step.

Supports:
  - Explicit difficulty selection via reset(difficulty=...)
  - Explicit problem selection via reset(problem_id=...)
  - Improvement-based reward shaping (delta reward between attempts)
  - Configurable max attempts
"""

from __future__ import annotations

import random
import uuid
from typing import Any, List, Optional

from openenv.core.env_server import Environment

from ..models import ApiDesignAction, ApiDesignObservation, ApiDesignState
from .grader import grade
from .problems import PROBLEMS, Problem, get_problem, get_problems_by_difficulty


class ApiDesignEnvironment(Environment):
    """
    RL environment where an agent designs REST API endpoints.

    Each episode presents a problem (set of functional requirements).
    The agent submits endpoint designs and receives multi-dimensional
    feedback with partial credit at every step.

    Reset kwargs
    ------------
    difficulty : str, optional
        Filter problems to "easy", "medium", or "hard".
    problem_id : str, optional
        Select a specific problem by ID (overrides difficulty).
    max_attempts : int, optional
        Override the default maximum number of attempts (default 5).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 5

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._state = ApiDesignState()
        self._problem: Optional[Problem] = None
        self._attempt = 0
        self._best_score = 0.0
        self._prev_score = 0.0
        self._max_attempts = self.MAX_ATTEMPTS

    # ── reset ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ApiDesignObservation:
        if seed is not None:
            random.seed(seed)

        self._max_attempts = int(kwargs.get("max_attempts", self.MAX_ATTEMPTS))

        # Problem selection: explicit id > difficulty filter > random
        problem_id: Optional[str] = kwargs.get("problem_id")
        difficulty: Optional[str] = kwargs.get("difficulty")

        if problem_id is not None:
            self._problem = get_problem(problem_id)
        elif difficulty is not None:
            pool = get_problems_by_difficulty(difficulty)
            if not pool:
                raise ValueError(f"No problems for difficulty={difficulty!r}")
            self._problem = random.choice(pool)
        else:
            self._problem = random.choice(PROBLEMS)

        self._attempt = 0
        self._best_score = 0.0
        self._prev_score = 0.0

        self._state = ApiDesignState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            problem_id=self._problem["id"],
            difficulty=self._problem["difficulty"],
            best_score=0.0,
            max_attempts=self._max_attempts,
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
                f"You have {self._max_attempts} attempts.",
            ],
            attempt_number=0,
            max_attempts=self._max_attempts,
            total_score=None,
        )

    # ── step ─────────────────────────────────────────────────────────

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
                max_attempts=self._max_attempts,
                total_score=0.0,
            )

        self._attempt += 1
        self._state.step_count += 1

        submitted = [ep.model_dump(exclude={"metadata"}) for ep in action.endpoints]
        result = grade(submitted, self._problem["ground_truth"])

        total: float = result["total"]
        self._best_score = max(self._best_score, total)
        self._state.best_score = self._best_score

        # Reward shaping: base score + improvement bonus
        improvement = max(0.0, total - self._prev_score)
        shaped_reward = round(total + 0.2 * improvement, 4)
        self._prev_score = total

        perfect = total >= 0.95
        exhausted = self._attempt >= self._max_attempts
        done = perfect or exhausted

        if perfect:
            result["suggestions"] = ["Excellent API design! Near-perfect score."]
        elif done:
            result["suggestions"].insert(
                0, f"Final attempt reached. Best score: {self._best_score:.2f}"
            )

        return ApiDesignObservation(
            done=done,
            reward=shaped_reward,
            requirements=self._problem["description"],
            constraints=self._problem["constraints"],
            feedback=result["scores"],
            suggestions=result["suggestions"],
            attempt_number=self._attempt,
            max_attempts=self._max_attempts,
            total_score=total,
        )

    # ── state ────────────────────────────────────────────────────────

    @property
    def state(self) -> ApiDesignState:
        return self._state
