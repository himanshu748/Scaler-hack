#!/usr/bin/env python3
"""
Baseline inference script for the API Design RL Environment.

Runs a heuristic agent (no API key needed) against the environment
and produces reproducible baseline scores. Optionally uses an
OpenAI-compatible LLM if OPENAI_API_KEY is set.

Usage:
    python inference.py
    OPENAI_API_KEY=sk-... python inference.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api_design_env.models import ApiDesignAction, EndpointSpec
    from api_design_env.server.environment import ApiDesignEnvironment
    from api_design_env.server.problems import PROBLEMS, get_problems_by_difficulty
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
    from api_design_env.models import ApiDesignAction, EndpointSpec
    from api_design_env.server.environment import ApiDesignEnvironment
    from api_design_env.server.problems import PROBLEMS, get_problems_by_difficulty


# ── Heuristic agent (no API key needed) ─────────────────────────────

def heuristic_agent(requirements: str, constraints: List[str]) -> List[EndpointSpec]:
    """Parse requirements text to produce a reasonable API design."""
    all_text = requirements + " " + " ".join(constraints)

    stop = {
        "this", "that", "with", "from", "have", "does", "also", "uses",
        "status", "items", "types", "rules", "access", "dates", "names",
        "levels", "limits", "steps", "roles", "images", "prices",
    }
    resource_candidates: List[str] = []

    for c in constraints:
        m = re.search(r"(?:crud for|manage|list|support)\s+(\w+)", c, re.I)
        if m:
            word = m.group(1).lower().rstrip(".")
            if word not in stop and len(word) > 2:
                if not word.endswith("s"):
                    word += "s"
                resource_candidates.append(word)

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

        endpoints.append(EndpointSpec(
            method="GET", path=base,
            description=f"List {resource}", status_code=200,
            request_body={}, response_body={resource: "list", "count": "int"},
            query_params=["limit", "offset"],
        ))
        endpoints.append(EndpointSpec(
            method="GET", path=f"{base}/{{id}}",
            description=f"Get {singular}", status_code=200,
            request_body={}, response_body={"id": "int", "name": "string"},
            query_params=[],
        ))
        endpoints.append(EndpointSpec(
            method="POST", path=base,
            description=f"Create {singular}", status_code=201,
            request_body={"name": "string"},
            response_body={"id": "int", "name": "string"},
            query_params=[],
        ))
        endpoints.append(EndpointSpec(
            method="PUT", path=f"{base}/{{id}}",
            description=f"Update {singular}", status_code=200,
            request_body={"name": "string"},
            response_body={"id": "int", "name": "string"},
            query_params=[],
        ))
        endpoints.append(EndpointSpec(
            method="DELETE", path=f"{base}/{{id}}",
            description=f"Delete {singular}", status_code=204,
            request_body={}, response_body={},
            query_params=[],
        ))

    add_crud(primary)
    for r in resources[1:4]:
        add_crud(r, parent=primary)

    return endpoints


# ── LLM agent (optional, needs OPENAI_API_KEY) ──────────────────────

def llm_agent(requirements: str, constraints: List[str]) -> Optional[List[EndpointSpec]]:
    """Call OpenAI API. Returns None on any failure."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        client = OpenAI(api_key=key)
        user_msg = (
            f"Requirements:\n{requirements}\n\nConstraints:\n"
            + "\n".join(f"- {c}" for c in constraints)
            + "\n\nReturn a JSON array of endpoint specs. Each: "
            + '{"method","path","description","request_body","response_body","status_code","query_params"}.'
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert API designer. Return ONLY a JSON array."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        parsed = json.loads(text.strip())
        if not isinstance(parsed, list):
            return None
        return [
            EndpointSpec(
                method=ep.get("method", "GET"),
                path=ep.get("path", "/"),
                description=ep.get("description", ""),
                request_body=ep.get("request_body", {}),
                response_body=ep.get("response_body", {}),
                status_code=ep.get("status_code", 200),
                query_params=ep.get("query_params", []),
            )
            for ep in parsed
        ]
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


# ── Main runner ──────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  API Design RL Environment -- Baseline Inference")
    print("=" * 70)

    use_llm = bool(os.environ.get("OPENAI_API_KEY"))
    agent_name = "gpt-4o-mini" if use_llm else "heuristic"
    print(f"\n  Agent: {agent_name}")

    env = ApiDesignEnvironment()
    n_episodes = int(os.environ.get("N_EPISODES", "5"))
    episodes: List[Dict[str, Any]] = []

    for i in range(n_episodes):
        try:
            problem = PROBLEMS[i % len(PROBLEMS)]
            pid = problem["id"]
            diff = problem["difficulty"]
            print(f"  [{i+1}/{n_episodes}] {pid} ({diff})...", end=" ", flush=True)

            obs = env.reset(problem_id=pid)

            if use_llm:
                llm_result = llm_agent(obs.requirements, obs.constraints)
                if llm_result:
                    action = ApiDesignAction(endpoints=llm_result)
                else:
                    action = ApiDesignAction(endpoints=heuristic_agent(obs.requirements, obs.constraints))
            else:
                action = ApiDesignAction(endpoints=heuristic_agent(obs.requirements, obs.constraints))

            obs = env.step(action)
            score = obs.total_score or 0.0
            print(f"score={score:.4f}  ({len(action.endpoints)} endpoints)")
            episodes.append({
                "problem_id": pid,
                "difficulty": diff,
                "score": score,
                "n_endpoints": len(action.endpoints),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            episodes.append({"problem_id": "unknown", "score": 0.0, "error": str(e)})

    scores = [e.get("score", 0.0) or 0.0 for e in episodes]

    print()
    print("  " + "-" * 50)
    print(f"  Episodes:  {len(episodes)}")
    if scores:
        print(f"  Mean:      {statistics.mean(scores):.4f}")
        print(f"  Min:       {min(scores):.4f}")
        print(f"  Max:       {max(scores):.4f}")
        if len(scores) > 1:
            print(f"  Stdev:     {statistics.stdev(scores):.4f}")
    print(f"  Agent:     {agent_name}")
    print("  " + "-" * 50)

    summary = {
        "agent": agent_name,
        "episodes": len(episodes),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "scores": [round(s, 4) for s in scores],
    }
    print()
    print(json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        print(json.dumps({"agent": "error", "episodes": 0, "mean_score": 0.0, "error": str(e)}))
