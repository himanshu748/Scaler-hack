---
title: API Design Environment
emoji: 🏗️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# API Design RL Environment

An OpenEnv RL environment for training AI agents to design REST APIs.

## How It Works

Given functional requirements (e.g. "Design a REST API for a todo-list app"), the agent submits endpoint specifications and receives multi-dimensional feedback:

| Dimension | Weight | What It Checks |
|---|---|---|
| Completeness | 0.30 | Coverage of required operations |
| RESTful Conventions | 0.25 | Correct methods, plural nouns, nesting |
| Schema Quality | 0.20 | Request/response body quality |
| Consistency | 0.15 | Naming and format uniformity |
| HTTP Semantics | 0.10 | Method safety, idempotency, status codes |

## Usage

```python
from api_design_env import ApiDesignEnv, ApiDesignAction, EndpointSpec

# Connect to the running Space
with ApiDesignEnv(base_url="https://<username>-api-design-env.hf.space").sync() as env:
    result = env.reset()
    print(result.observation.requirements)
    print(result.observation.constraints)

    action = ApiDesignAction(endpoints=[
        EndpointSpec(method="GET", path="/todos", description="List todos",
                     status_code=200, query_params=["completed", "limit"]),
        EndpointSpec(method="POST", path="/todos", description="Create todo",
                     status_code=201, request_body={"title": "string"}),
    ])
    result = env.step(action)
    print(f"Score: {result.observation.total_score}")
    print(f"Feedback: {result.observation.feedback}")
    print(f"Suggestions: {result.observation.suggestions}")
```

## Problem Bank

10+ curated problems across 3 difficulty levels:
- **Easy**: Todo CRUD, Bookmark Manager, Notes App, Contacts
- **Medium**: E-Commerce Catalog, Blog Platform, Event Management, Kanban Board
- **Hard**: Multi-Tenant SaaS with RBAC, Cloud File Storage, Real-Time Messaging

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face
- Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
