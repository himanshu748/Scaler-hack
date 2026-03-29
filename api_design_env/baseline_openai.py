#!/usr/bin/env python3
"""
OpenAI API baseline for the API Design RL Environment.

Runs gpt-4o-mini (or any OpenAI-compatible model) against all problems
and reports reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m api_design_env.baseline_openai

    # Custom model
    python -m api_design_env.baseline_openai --model gpt-4o

    # Single difficulty
    python -m api_design_env.baseline_openai --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

from .models import ApiDesignAction, EndpointSpec
from .server.environment import ApiDesignEnvironment
from .server.problems import PROBLEMS, get_problems_by_difficulty

SYSTEM_PROMPT = """\
You are an expert API designer. You will be given functional requirements for a
REST API and must return a JSON array of endpoint specifications.

Each endpoint object must have these fields:
- "method": HTTP method (GET, POST, PUT, PATCH, DELETE)
- "path": URL path with path params in curly braces, e.g. /users/{id}/posts
- "description": brief description of what this endpoint does
- "request_body": object mapping field names to type hints (empty {} for GET/DELETE)
- "response_body": object mapping field names to type hints
- "status_code": expected success status code (200, 201, 204, etc.)
- "query_params": array of supported query parameter names

Follow RESTful conventions:
- Use plural nouns for resources (e.g. /todos not /todo)
- No verbs in paths (use HTTP methods instead)
- Use 201 for POST creation, 204 for DELETE
- Nest sub-resources under parents (e.g. /posts/{post_id}/comments)

Return ONLY the JSON array, no markdown fences, no explanation.\
"""


def call_openai(
    requirements: str,
    constraints: List[str],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Call OpenAI API and parse the response into endpoint dicts."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Get a key at https://platform.openai.com/api-keys")
        sys.exit(1)

    client = OpenAI(api_key=key)

    user_msg = f"""Requirements:
{requirements}

Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Return the JSON array of endpoint specifications."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=4096,
    )

    text = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    try:
        endpoints = json.loads(text)
    except json.JSONDecodeError:
        print(f"  Warning: failed to parse LLM response as JSON")
        return []

    if not isinstance(endpoints, list):
        return []
    return endpoints


def parse_endpoints(raw: List[Dict[str, Any]]) -> ApiDesignAction:
    """Convert raw dicts from the LLM into a typed ApiDesignAction."""
    specs = []
    for ep in raw:
        try:
            specs.append(
                EndpointSpec(
                    method=ep.get("method", "GET"),
                    path=ep.get("path", "/"),
                    description=ep.get("description", ""),
                    request_body=ep.get("request_body", {}),
                    response_body=ep.get("response_body", {}),
                    status_code=ep.get("status_code", 200),
                    query_params=ep.get("query_params", []),
                )
            )
        except Exception:
            continue
    return ApiDesignAction(endpoints=specs)


def run_openai_baseline(
    model: str = "gpt-4o-mini",
    difficulty_filter: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the OpenAI agent against all problems and return results."""
    env = ApiDesignEnvironment()

    difficulties = (
        [difficulty_filter] if difficulty_filter else ["easy", "medium", "hard"]
    )

    episodes = []
    for diff in difficulties:
        problems = get_problems_by_difficulty(diff)
        for problem in problems:
            pid = problem["id"]
            print(f"  Running {model} on {pid} ({diff})...", end=" ", flush=True)

            obs = env.reset(problem_id=pid)
            raw = call_openai(
                obs.requirements, obs.constraints, model=model, api_key=api_key
            )
            action = parse_endpoints(raw)
            obs = env.step(action)

            print(f"score={obs.total_score:.4f}")
            episodes.append(
                {
                    "problem_id": pid,
                    "difficulty": diff,
                    "score": obs.total_score,
                    "feedback": obs.feedback,
                    "n_endpoints_submitted": len(action.endpoints),
                }
            )

    scores = [e["score"] for e in episodes]
    by_diff: Dict[str, List[float]] = {}
    for e in episodes:
        by_diff.setdefault(e["difficulty"], []).append(e["score"])

    return {
        "model": model,
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "by_difficulty": {
            d: round(statistics.mean(s), 4) for d, s in sorted(by_diff.items())
        },
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI API baseline for API Design RL Environment"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run only one difficulty tier",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  API Design RL Environment -- OpenAI Baseline ({args.model})")
    print("=" * 70)
    print()

    result = run_openai_baseline(
        model=args.model, difficulty_filter=args.difficulty
    )

    if args.json:
        compact = {k: v for k, v in result.items() if k != "episodes"}
        print(json.dumps(compact, indent=2))
        return

    print()
    print(f"Model: {result['model']}")
    print(f"Mean:  {result['mean_score']:.4f}")
    print(f"Min:   {result['min_score']:.4f}")
    print(f"Max:   {result['max_score']:.4f}")
    print()
    print("By difficulty:")
    for d, s in result["by_difficulty"].items():
        print(f"  {d:<8} {s:.4f}")
    print()
    print("Per-problem:")
    print(f"  {'Problem':<25} {'Diff':<8} {'Score':>8} {'Endpoints':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    for ep in result["episodes"]:
        print(
            f"  {ep['problem_id']:<25} {ep['difficulty']:<8} "
            f"{ep['score']:>8.4f} {ep['n_endpoints_submitted']:>10}"
        )


if __name__ == "__main__":
    main()
