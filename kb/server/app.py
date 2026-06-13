"""create_app(kb) factory — assembles routes + startup into a Starlette app.

Usage:
    from kb.server import create_app
    app = create_app(kb)
    uvicorn.run(app, host=host, port=port, log_level="warning")
"""

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute

from .bridge import bridge_messages, bridge_agents, bridge_watch
from .live import make_live_handlers
from .routes import make_web_handlers


def create_app(kb) -> Starlette:
    """Build and return the Starlette ASGI app bound to the given KnowledgeBase.

    All route handlers capture `kb` via closure through the make_*_handlers()
    factories, mirroring the original inline-closure design in main().
    """
    index, search_page, finding_page = make_web_handlers(kb)
    ws_updates, on_startup = make_live_handlers(kb)

    routes = [
        Route("/", index),
        Route("/search", search_page),
        Route("/finding/{id:path}", finding_page),
        WebSocketRoute("/ws", ws_updates),
        Route("/bridge/messages", bridge_messages),
        Route("/bridge/agents", bridge_agents),
        Route("/bridge/watch", bridge_watch),
    ]
    return Starlette(routes=routes, on_startup=[on_startup])
