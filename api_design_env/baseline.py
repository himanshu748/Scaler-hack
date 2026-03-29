#!/usr/bin/env python3
"""
Baseline inference script for the API Design RL Environment.

Demonstrates three agent policies (random, heuristic, oracle) running
against every difficulty tier and reports reproducible scores.

Usage:
    # Run all baselines locally (no server needed)
    python -m api_design_env.baseline

    # Run against a live HF Space
    python -m api_design_env.baseline --url https://himanshukumarjha-api-design-env.hf.space

    # Run a single difficulty
    python -m api_design_env.baseline --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from typing import Any, Dict, List, Optional

from .models import ApiDesignAction, EndpointSpec
from .server.environment import ApiDesignEnvironment
from .server.problems import PROBLEMS, get_problems_by_difficulty

# ── Agent policies ──────────────────────────────────────────────────


def random_agent(obs_data: Dict[str, Any]) -> ApiDesignAction:
    """Generates random endpoints -- represents an untrained agent."""
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    bad_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "FETCH", "REMOVE"]
    resources = ["items", "users", "data", "things", "records"]
    bad_paths = ["/getItems", "/createUser", "/deleteAll", "/fetchData"]
    n_endpoints = random.randint(1, 8)
    endpoints = []
    for _ in range(n_endpoints):
        if random.random() < 0.3:
            # Produce bad patterns sometimes
            method = random.choice(bad_methods)
            path = random.choice(bad_paths)
            status = random.choice([200, 201, 204, 400, 500])
        else:
            method = random.choice(methods)
            resource = random.choice(resources)
            path = f"/{resource}" if random.random() > 0.5 else f"/{resource}/{{id}}"
            status = {"GET": 200, "POST": 201, "DELETE": 204}.get(method, 200)
        endpoints.append(
            EndpointSpec(
                method=method,
                path=path,
                description=f"{method} {path}" if random.random() > 0.3 else "",
                status_code=status,
            )
        )
    return ApiDesignAction(endpoints=endpoints)


def heuristic_agent(obs_data: Dict[str, Any]) -> ApiDesignAction:
    """
    Parses the requirements text and constraints to produce a reasonable
    API design.  Represents a moderately capable agent.

    Strategy:
      1. Extract resource nouns from constraints (more targeted than description).
      2. Generate standard CRUD for each identified resource.
      3. Wire up sub-resources when constraints mention relationships.
    """
    requirements: str = obs_data.get("requirements", "")
    constraints: List[str] = obs_data.get("constraints", [])
    all_text = requirements + " " + " ".join(constraints)

    # -- Extract resource names from constraints and requirements --
    import re

    stop = {
        "this", "that", "with", "from", "have", "does", "also", "uses",
        "status", "items", "types", "rules", "access", "dates", "names",
        "levels", "limits", "steps", "roles", "images", "prices",
    }
    resource_candidates: List[str] = []

    # Look for "CRUD for <resource>" or "Manage <resource>" patterns
    for c in constraints:
        m = re.search(r"(?:crud for|manage|list|support)\s+(\w+)", c, re.I)
        if m:
            word = m.group(1).lower().rstrip(".")
            if word not in stop and len(word) > 2:
                if not word.endswith("s"):
                    word += "s"
                resource_candidates.append(word)

    # Fallback: plural nouns from all text
    if not resource_candidates:
        for word in re.findall(r"\b([a-z]{4,}s)\b", all_text.lower()):
            if word not in stop:
                resource_candidates.append(word)

    seen: set = set()
    resources: List[str] = []
    for r in resource_candidates:
        if r not in seen:
            seen.add(r)
            resources.append(r)
    if not resources:
        resources = ["items"]

    primary = resources[0]
    endpoints: List[EndpointSpec] = []

    def add_crud(resource: str, parent: str = "") -> None:
        prefix = f"/{parent}/{{id}}" if parent else ""
        base = f"{prefix}/{resource}"
        singular = resource.rstrip("s") if resource.endswith("s") else resource

        endpoints.append(
            EndpointSpec(
                method="GET", path=base,
                description=f"List {resource}",
                status_code=200,
                query_params=["limit", "offset"],
                response_body={resource: "list", "count": "int"},
            )
        )
        endpoints.append(
            EndpointSpec(
                method="GET", path=f"{base}/{{id}}",
                description=f"Get {singular}",
                status_code=200,
                response_body={"id": "int", "name": "string"},
            )
        )
        endpoints.append(
            EndpointSpec(
                method="POST", path=base,
                description=f"Create {singular}",
                status_code=201,
                request_body={"name": "string"},
                response_body={"id": "int", "name": "string"},
            )
        )
        endpoints.append(
            EndpointSpec(
                method="PUT", path=f"{base}/{{id}}",
                description=f"Update {singular}",
                status_code=200,
                request_body={"name": "string"},
                response_body={"id": "int", "name": "string"},
            )
        )
        endpoints.append(
            EndpointSpec(
                method="DELETE", path=f"{base}/{{id}}",
                description=f"Delete {singular}",
                status_code=204,
            )
        )

    # CRUD for primary resource
    add_crud(primary)

    # Sub-resources (nest under primary)
    for r in resources[1:4]:
        is_sub = any(
            r in c.lower() or r.rstrip("s") in c.lower() for c in constraints
        )
        if is_sub:
            add_crud(r, parent=primary)

    return ApiDesignAction(endpoints=endpoints)


def oracle_agent(problem_ground_truth: List[Dict[str, Any]]) -> ApiDesignAction:
    """
    Uses the ground-truth solution directly.  Represents the theoretical
    maximum score.  Used to verify grader correctness.
    """
    endpoints = []
    for ep in problem_ground_truth:
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
    return ApiDesignAction(endpoints=endpoints)


# ── Runner ──────────────────────────────────────────────────────────


def run_episode(
    env: ApiDesignEnvironment,
    agent_fn,
    problem_id: str,
    seed: int = 0,
    oracle_gt: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run a single episode and return metrics."""
    obs = env.reset(seed=seed, problem_id=problem_id)
    obs_data = obs.model_dump()
    total_reward = 0.0
    steps = 0

    while not obs.done:
        if oracle_gt is not None:
            action = oracle_agent(oracle_gt)
        else:
            action = agent_fn(obs_data)
        obs = env.step(action)
        obs_data = obs.model_dump()
        total_reward += obs.reward or 0.0
        steps += 1

    return {
        "problem_id": problem_id,
        "difficulty": env.state.difficulty,
        "best_score": env.state.best_score,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "final_feedback": obs.feedback,
    }


def run_baseline(
    difficulty_filter: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run all three agents across the problem bank and return results."""
    random.seed(seed)
    env = ApiDesignEnvironment()

    difficulties = (
        [difficulty_filter] if difficulty_filter else ["easy", "medium", "hard"]
    )

    results: Dict[str, List[Dict]] = {"random": [], "heuristic": [], "oracle": []}

    for diff in difficulties:
        problems = get_problems_by_difficulty(diff)
        for problem in problems:
            pid = problem["id"]

            # Random agent
            r = run_episode(env, random_agent, pid, seed=seed)
            results["random"].append(r)

            # Heuristic agent
            r = run_episode(env, heuristic_agent, pid, seed=seed)
            results["heuristic"].append(r)

            # Oracle agent
            r = run_episode(
                env, None, pid, seed=seed, oracle_gt=problem["ground_truth"]
            )
            results["oracle"].append(r)

    # Aggregate
    summary: Dict[str, Any] = {}
    for agent_name, episodes in results.items():
        scores = [e["best_score"] for e in episodes]
        by_diff: Dict[str, List[float]] = {}
        for e in episodes:
            by_diff.setdefault(e["difficulty"], []).append(e["best_score"])

        summary[agent_name] = {
            "mean_score": round(statistics.mean(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "by_difficulty": {
                d: round(statistics.mean(s), 4) for d, s in sorted(by_diff.items())
            },
            "episodes": episodes,
        }

    return summary


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference for API Design RL Environment"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run only one difficulty tier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of table"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  API Design RL Environment -- Baseline Evaluation")
    print("=" * 70)
    print()

    summary = run_baseline(difficulty_filter=args.difficulty, seed=args.seed)

    if args.json:
        # Strip full episode details for compact output
        compact = {}
        for agent, data in summary.items():
            compact[agent] = {k: v for k, v in data.items() if k != "episodes"}
        print(json.dumps(compact, indent=2))
        return

    # Pretty table
    header = f"{'Agent':<12} {'Mean':>8} {'Min':>8} {'Max':>8}"
    diff_cols = ""
    diffs = sorted(
        {d for a in summary.values() for d in a["by_difficulty"]},
        key=lambda x: ["easy", "medium", "hard"].index(x),
    )
    for d in diffs:
        header += f" {d.capitalize():>8}"
    print(header)
    print("-" * len(header))

    for agent_name in ["random", "heuristic", "oracle"]:
        data = summary[agent_name]
        row = (
            f"{agent_name:<12} "
            f"{data['mean_score']:>8.4f} "
            f"{data['min_score']:>8.4f} "
            f"{data['max_score']:>8.4f}"
        )
        for d in diffs:
            row += f" {data['by_difficulty'].get(d, 0.0):>8.4f}"
        print(row)

    print()
    print("Agents:")
    print("  random     - Generates random endpoints (untrained baseline)")
    print("  heuristic  - Parses requirements text to guess CRUD endpoints")
    print("  oracle     - Submits the ground-truth solution (upper bound)")
    print()

    # Per-problem detail for heuristic
    print("Heuristic agent per-problem scores:")
    print(f"  {'Problem':<25} {'Diff':<8} {'Score':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for ep in summary["heuristic"]["episodes"]:
        print(f"  {ep['problem_id']:<25} {ep['difficulty']:<8} {ep['best_score']:>8.4f}")


if __name__ == "__main__":
    main()
