from __future__ import annotations

"""Runtime entrypoint to launch the FastAPI chatbot server."""

import uvicorn

from app.config import settings


def main() -> None:
    """Start uvicorn server with environment-driven host/port."""

    uvicorn.run("api:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
