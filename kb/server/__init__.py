"""kb/server — HTTP/SSE/WebSocket server for the knowledge base.

Public API:
    create_app(kb) -> Starlette   build and return the ASGI app
"""

from .app import create_app

__all__ = ["create_app"]
