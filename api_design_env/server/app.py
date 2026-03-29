"""FastAPI application for the API Design environment."""

import os

from openenv.core.env_server import create_fastapi_app

from ..models import ApiDesignAction, ApiDesignObservation
from .environment import ApiDesignEnvironment

app = create_fastapi_app(
    env=ApiDesignEnvironment,
    action_cls=ApiDesignAction,
    observation_cls=ApiDesignObservation,
)


def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "4"))
    uvicorn.run(
        "api_design_env.server.app:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
