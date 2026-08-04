"""Compatibility entry point for the FastAPI web demo."""

import os

import uvicorn

from web_app import app


if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8001")),
        reload=False,
    )


__all__ = ["app"]
