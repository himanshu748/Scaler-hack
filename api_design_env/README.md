---
title: API Design Environment
emoji: "\U0001F3D7\uFE0F"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
pinned: false
---

# API Design RL Environment

A production-relevant OpenEnv environment where an AI agent learns to **design
REST APIs from functional requirements**.  Given a natural-language spec (e.g.
"Design a REST API for an e-commerce product catalog with filtering, variants,
and categories"), the agent submits endpoint definitions and receives
multi-dimensional, partial-credit feedback until it converges on a correct
design.

## Motivation

API design is a routine engineering task with clear, objectively verifiable
quality criteria: correct HTTP methods, RESTful path conventions, complete
coverage of requirements, consistent naming, and proper status codes.  Unlike
toy tasks these properties map directly to real production code reviews, making
this environment useful for training agents that assist with software
engineering workflows.

---

## Action Space

The agent submits an `ApiDesignAction` containing a list of `EndpointSpec`
objects.

```python
class EndpointSpec(Action):
    method: str            # HTTP method: GET, POST, PUT, PATCH, DELETE
    path: str              # URL path, e.g. /users/{id}/posts
    description: str       # What this endpoint does (default "")
    request_body: dict     # Request body schema {field: type} (default {})
    response_body: dict    # Response body schema {field: type} (default {})
    status_code: int       # Expected success status code (default 200)
    query_params: list[str]# Supported query parameters (default [])

class ApiDesignAction(Action):
    endpoints: list[EndpointSpec]
```

## Observation Space

After each `step()` the agent receives an `ApiDesignObservation`:

```python
class ApiDesignObservation(Observation):
    # Inherited from Observation base
    done: bool                       # Episode finished?
    reward: float | None             # Shaped reward signal (0.0-1.2)

    # Environment-specific
    requirements: str                # Natural-language problem description
    constraints: list[str]           # Specific constraints to satisfy
    feedback: dict[str, float] | None# Per-dimension scores (5 axes)
    suggestions: list[str]           # Actionable improvement hints
    attempt_number: int              # Current attempt (1-indexed)
    max_attempts: int                # Maximum allowed attempts
    total_score: float | None        # Weighted total score (0.0-1.0)
```

## State

```python
class ApiDesignState(State):
    episode_id: str     # Unique episode identifier
    step_count: int     # Steps taken so far
    problem_id: str     # Current problem identifier
    difficulty: str     # "easy" | "medium" | "hard"
    best_score: float   # Best total_score achieved this episode
    max_attempts: int   # Configured max attempts
```

---

## Reward Function

The reward is **not binary**.  Every `step()` returns a composite signal built
from five programmatic grading dimensions:

| Dimension | Weight | What It Checks |
|---|---|---|
| **Completeness** | 0.30 | Does the design cover all required operations? |
| **RESTful Conventions** | 0.25 | Correct HTTP methods, plural nouns, no verbs in paths |
| **Schema Quality** | 0.20 | Request/response body field coverage, query params |
| **Consistency** | 0.15 | Naming uniformity, trailing slashes, descriptions |
| **HTTP Semantics** | 0.10 | Method safety, idempotency, status code correctness |

**Partial progress signals:**

- Each dimension scores 0.0-1.0 independently -- partial credit within each.
- `total_score = weighted sum` (0.0-1.0).
- **Improvement bonus:** `reward = total_score + 0.2 * max(0, score - prev_score)`.
  First good attempt gets a boost; subsequent regressions do not.
- **Penalty signals:** empty submissions receive 0.0 across all dimensions.
  Endpoints with invalid HTTP methods, verbs in paths, or wrong status codes
  are penalised in the restful_conventions and http_semantics dimensions.

The `suggestions` list provides natural-language hints (e.g. "Missing endpoint:
DELETE /todos/{id}", "Avoid verbs in path: /getUser") that a language-model
agent can use to iteratively improve.

---

## Tasks (3 Difficulty Tiers, 11 Problems)

### Easy (4 problems)

| ID | Title | Endpoints | Description |
|---|---|---|---|
| `todo_crud` | Todo List CRUD | 5 | Standard CRUD for a todo-list app |
| `bookmark_manager` | Bookmark Manager | 6 | Save, organise, search bookmarks with tags |
| `notes_app` | Notes App | 5 | Create/read/update/delete notes with timestamps |
| `contacts_api` | Contacts API | 5 | Manage personal contacts with search |

### Medium (4 problems)

| ID | Title | Endpoints | Description |
|---|---|---|---|
| `ecommerce_products` | E-Commerce Catalog | 9 | Products, categories, variants with filtering |
| `blog_platform` | Blog Platform | 10 | Posts with drafts, comments, tags |
| `event_management` | Event Management | 10 | Events, registrations, ticket types |
| `task_board` | Kanban Board | 10 | Boards, columns, cards, assignees |

### Hard (3 problems)

| ID | Title | Endpoints | Description |
|---|---|---|---|
| `multi_tenant_saas` | Multi-Tenant SaaS | 10 | Tenant-scoped users, RBAC, invitations, settings |
| `file_storage_api` | Cloud File Storage | 11 | Folders, files, versions, sharing, search |
| `messaging_platform` | Messaging Platform | 14 | Channels, messages, threads, reactions, pins |

Every problem has a deterministic ground-truth solution.  The grader compares
the agent's submission against it structurally (path patterns, methods, schema
fields) -- no LLM is needed for scoring.

---

## Setup

### Install

```bash
pip install openenv-core
git clone https://github.com/himanshu748/Scaler-hack.git
cd Scaler-hack
```

### Run locally (no Docker)

```bash
PYTHONPATH=. uvicorn api_design_env.server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t api-design-env -f api_design_env/Dockerfile api_design_env/
docker run -p 8000:8000 api-design-env
```

### Validate

```bash
cd api_design_env && openenv validate
# [OK] api_design: Ready for multi-mode deployment
```

### Deploy to HF Spaces

```bash
cd api_design_env && openenv push --repo-id <username>/api-design-env
```

---

## Usage

### Python (direct, no server)

```python
from api_design_env import ApiDesignEnv, ApiDesignAction, EndpointSpec
from api_design_env.server.environment import ApiDesignEnvironment

env = ApiDesignEnvironment()

# Reset with difficulty or problem selection
obs = env.reset(seed=42, difficulty="easy")
# obs = env.reset(problem_id="todo_crud")

print(obs.requirements)   # "Design a REST API for a simple todo-list..."
print(obs.constraints)    # ["Support listing all todos...", ...]

# Submit a design
action = ApiDesignAction(endpoints=[
    EndpointSpec(method="GET", path="/todos", description="List todos",
                 status_code=200, query_params=["completed", "limit"]),
    EndpointSpec(method="POST", path="/todos", description="Create todo",
                 status_code=201, request_body={"title": "string"}),
])
obs = env.step(action)

print(obs.total_score)   # 0.45 (partial credit)
print(obs.feedback)      # {"completeness": 0.4, "restful_conventions": 1.0, ...}
print(obs.suggestions)   # ["Missing endpoint: DELETE /todos/{id}", ...]
```

### WebSocket client (against running server)

```python
from api_design_env import ApiDesignEnv, ApiDesignAction, EndpointSpec

with ApiDesignEnv(base_url="https://himanshukumarjha-api-design-env.hf.space").sync() as env:
    result = env.reset()
    print(result.observation.requirements)
    result = env.step(ApiDesignAction(endpoints=[...]))
    print(result.observation.total_score)
```

---

## Baseline Scores

Run the baseline evaluation (deterministic, seed=42):

```bash
python -m api_design_env           # Pretty table
python -m api_design_env --json    # Machine-readable JSON
```

### Results

| Agent | Mean | Min | Max | Easy | Medium | Hard |
|---|---|---|---|---|---|---|
| **random** | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 |
| **heuristic** | 0.64 | 0.43 | 0.96 | 0.57 | 0.69 | 0.66 |
| **oracle** | 0.99 | 0.96 | 1.00 | 1.00 | 1.00 | 0.98 |

- **random** -- generates structurally valid but content-irrelevant endpoints.
- **heuristic** -- parses requirement text to extract resource names and
  generates standard CRUD.  No LLM.
- **oracle** -- submits the ground-truth solution (theoretical upper bound).

### OpenAI baseline

```bash
export OPENAI_API_KEY=sk-...
python -m api_design_env.baseline_openai
```

Uses `gpt-4o-mini` to read requirements and produce endpoint designs.  See
`api_design_env/baseline_openai.py` for details.

---

## Endpoints (when deployed)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action, get observation |
| `/state` | GET | Current episode state |
| `/docs` | GET | OpenAPI documentation |
| `/web` | GET | Interactive web UI |
| `/ws` | WebSocket | Persistent session (used by Python client) |

---

## Project Structure

```
api_design_env/
├── __init__.py              # Public exports
├── __main__.py              # `python -m api_design_env` entry point
├── models.py                # Action, Observation, State (Pydantic)
├── client.py                # EnvClient subclass (WebSocket)
├── baseline.py              # Heuristic + random + oracle baselines
├── baseline_openai.py       # OpenAI API baseline (gpt-4o-mini)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package metadata
├── Dockerfile               # HF Spaces / production container
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── environment.py       # Core RL logic (reset/step/state)
    ├── grader.py            # Multi-dimensional scoring engine
    ├── problems.py          # 11 curated problems with ground truths
    ├── Dockerfile           # Alternative Dockerfile
    └── requirements.txt
tests/
└── test_api_design_env.py   # 14 unit tests
```

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face
- Built for the [Meta PyTorch OpenEnv Hackathon x SST](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
