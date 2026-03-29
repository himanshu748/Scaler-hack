"""Unit tests for the API Design RL environment."""

import pytest

from api_design_env.models import ApiDesignAction, EndpointSpec
from api_design_env.server.environment import ApiDesignEnvironment
from api_design_env.server.grader import grade
from api_design_env.server.problems import PROBLEMS, get_problem


class TestGrader:
    def test_perfect_score(self):
        problem = get_problem("todo_crud")
        result = grade(problem["ground_truth"], problem["ground_truth"])
        assert result["total"] == 1.0
        for v in result["scores"].values():
            assert v == 1.0

    def test_empty_submission(self):
        problem = get_problem("todo_crud")
        result = grade([], problem["ground_truth"])
        assert result["total"] == 0.0

    def test_partial_score(self):
        problem = get_problem("todo_crud")
        partial = [ep for ep in problem["ground_truth"] if ep["method"] == "GET"]
        result = grade(partial, problem["ground_truth"])
        assert 0.0 < result["total"] < 1.0
        assert result["scores"]["completeness"] < 1.0

    def test_all_problems_have_ground_truth(self):
        for p in PROBLEMS:
            assert len(p["ground_truth"]) > 0, f"{p['id']} has no ground truth"
            assert p["id"], "Problem missing id"
            assert p["difficulty"] in ("easy", "medium", "hard")
            assert len(p["constraints"]) > 0

    def test_all_problems_self_score_perfect(self):
        for p in PROBLEMS:
            result = grade(p["ground_truth"], p["ground_truth"])
            assert result["total"] >= 0.95, (
                f"{p['id']}: self-grade {result['total']} < 0.95. "
                f"Scores: {result['scores']}"
            )


class TestEnvironment:
    def test_reset_returns_observation(self):
        env = ApiDesignEnvironment()
        obs = env.reset(seed=1)
        assert obs.done is False
        assert obs.reward is None
        assert len(obs.requirements) > 0
        assert len(obs.constraints) > 0
        assert obs.attempt_number == 0

    def test_step_without_reset(self):
        env = ApiDesignEnvironment()
        action = ApiDesignAction(endpoints=[])
        obs = env.step(action)
        assert obs.done is True
        assert "reset()" in obs.suggestions[0]

    def test_step_increments_attempt(self):
        env = ApiDesignEnvironment()
        env.reset(seed=1)
        action = ApiDesignAction(
            endpoints=[
                EndpointSpec(method="GET", path="/test", description="test")
            ]
        )
        obs = env.step(action)
        assert obs.attempt_number == 1
        assert env.state.step_count == 1

    def test_episode_terminates_after_max_attempts(self):
        env = ApiDesignEnvironment()
        env.reset(seed=1)
        action = ApiDesignAction(
            endpoints=[
                EndpointSpec(method="GET", path="/test", description="test")
            ]
        )
        for i in range(env.MAX_ATTEMPTS):
            obs = env.step(action)

        assert obs.done is True
        assert obs.attempt_number == env.MAX_ATTEMPTS

    def test_perfect_submission_terminates_early(self):
        env = ApiDesignEnvironment()
        env.reset(seed=1)
        problem_id = env.state.problem_id
        problem = get_problem(problem_id)

        endpoints = []
        for ep in problem["ground_truth"]:
            endpoints.append(
                EndpointSpec(
                    method=ep["method"],
                    path=ep["path"],
                    description=ep.get("description", ""),
                    request_body=ep.get("request_body", {}),
                    response_body=ep.get("response_body", {}),
                    status_code=ep.get("status_code", 200),
                    query_params=ep.get("query_params", []),
                )
            )
        action = ApiDesignAction(endpoints=endpoints)
        obs = env.step(action)
        assert obs.done is True
        assert obs.total_score >= 0.95

    def test_state_tracks_best_score(self):
        env = ApiDesignEnvironment()
        env.reset(seed=1)
        problem_id = env.state.problem_id
        problem = get_problem(problem_id)

        # Bad submission
        action1 = ApiDesignAction(
            endpoints=[EndpointSpec(method="GET", path="/x", description="x")]
        )
        env.step(action1)
        score1 = env.state.best_score

        # Better submission
        first_ep = problem["ground_truth"][0]
        action2 = ApiDesignAction(
            endpoints=[
                EndpointSpec(
                    method=first_ep["method"],
                    path=first_ep["path"],
                    description=first_ep.get("description", ""),
                    request_body=first_ep.get("request_body", {}),
                    response_body=first_ep.get("response_body", {}),
                    status_code=first_ep.get("status_code", 200),
                    query_params=first_ep.get("query_params", []),
                )
            ]
        )
        env.step(action2)
        assert env.state.best_score >= score1


    def test_reset_with_difficulty(self):
        env = ApiDesignEnvironment()
        obs = env.reset(difficulty="easy")
        assert env.state.difficulty == "easy"
        obs = env.reset(difficulty="hard")
        assert env.state.difficulty == "hard"

    def test_reset_with_problem_id(self):
        env = ApiDesignEnvironment()
        obs = env.reset(problem_id="todo_crud")
        assert env.state.problem_id == "todo_crud"
        assert env.state.difficulty == "easy"

    def test_reset_invalid_difficulty_raises(self):
        env = ApiDesignEnvironment()
        with pytest.raises(ValueError):
            env.reset(difficulty="impossible")

    def test_reset_invalid_problem_id_raises(self):
        env = ApiDesignEnvironment()
        with pytest.raises(ValueError):
            env.reset(problem_id="nonexistent_problem")

    def test_custom_max_attempts(self):
        env = ApiDesignEnvironment()
        obs = env.reset(max_attempts=2)
        assert obs.max_attempts == 2
        action = ApiDesignAction(
            endpoints=[EndpointSpec(method="GET", path="/x", description="x")]
        )
        env.step(action)
        obs = env.step(action)
        assert obs.done is True
        assert obs.attempt_number == 2

    def test_improvement_reward_bonus(self):
        env = ApiDesignEnvironment()
        env.reset(seed=1, problem_id="todo_crud")
        problem = get_problem("todo_crud")

        # First step: bad submission
        bad = ApiDesignAction(
            endpoints=[EndpointSpec(method="GET", path="/x", description="x")]
        )
        obs1 = env.step(bad)
        r1 = obs1.reward

        # Second step: perfect submission (big improvement)
        eps = [
            EndpointSpec(
                method=ep["method"], path=ep["path"],
                description=ep.get("description", ""),
                request_body=ep.get("request_body", {}),
                response_body=ep.get("response_body", {}),
                status_code=ep.get("status_code", 200),
                query_params=ep.get("query_params", []),
            )
            for ep in problem["ground_truth"]
        ]
        obs2 = env.step(ApiDesignAction(endpoints=eps))
        # Shaped reward should be > raw total_score due to improvement bonus
        assert obs2.reward > obs2.total_score


class TestGraderPenalties:
    def test_empty_submission_penalty(self):
        problem = get_problem("todo_crud")
        result = grade([], problem["ground_truth"])
        assert result["penalty"] == 0.0
        assert result["total"] == 0.0

    def test_duplicate_endpoints_penalised(self):
        problem = get_problem("todo_crud")
        ep = problem["ground_truth"][0]
        duped = [ep, ep, ep]
        result = grade(duped, problem["ground_truth"])
        assert result["penalty"] < 1.0
        assert any("duplicate" in s.lower() for s in result["suggestions"])

    def test_over_submission_penalised(self):
        problem = get_problem("contacts_api")  # 5 endpoints
        bloated = problem["ground_truth"] * 5  # 25 endpoints
        result = grade(bloated, problem["ground_truth"])
        assert result["penalty"] < 1.0

    def test_no_penalty_for_good_submission(self):
        problem = get_problem("todo_crud")
        result = grade(problem["ground_truth"], problem["ground_truth"])
        assert result["penalty"] == 1.0


class TestProblems:
    def test_problem_count(self):
        assert len(PROBLEMS) >= 10

    def test_difficulty_distribution(self):
        easy = [p for p in PROBLEMS if p["difficulty"] == "easy"]
        medium = [p for p in PROBLEMS if p["difficulty"] == "medium"]
        hard = [p for p in PROBLEMS if p["difficulty"] == "hard"]
        assert len(easy) >= 3
        assert len(medium) >= 3
        assert len(hard) >= 2

    def test_unique_ids(self):
        ids = [p["id"] for p in PROBLEMS]
        assert len(ids) == len(set(ids))


class TestBaseline:
    def test_baseline_runs(self):
        from api_design_env.baseline import run_baseline
        summary = run_baseline(difficulty_filter="easy", seed=42)
        assert "random" in summary
        assert "heuristic" in summary
        assert "oracle" in summary
        assert summary["oracle"]["mean_score"] >= 0.95
        assert summary["random"]["mean_score"] < summary["oracle"]["mean_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
