"""create_app(kb) factory — assembles routes + startup into a Starlette app.

Usage:
    from kb.server import create_app
    app = create_app(kb)
    uvicorn.run(app, host=host, port=port, log_level="warning")
"""

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute

from .api import make_api_handlers
from .bridge import bridge_messages, bridge_agents, bridge_watch, bridge_send
from .federation import make_federation_handlers
from .live import make_live_handlers
from .routes import make_web_handlers


def create_app(kb) -> Starlette:
    """Build and return the Starlette ASGI app bound to the given KnowledgeBase.

    All route handlers capture `kb` via closure through the make_*_handlers()
    factories, mirroring the original inline-closure design in main().
    """
    index, search_page, finding_page = make_web_handlers(kb)
    ws_updates, on_startup = make_live_handlers(kb)
    kb_search, kb_recent, finding_get, issues_list, issue_get = make_api_handlers(kb)
    federated_search = make_federation_handlers(kb)

    routes = [
        Route("/", index),
        Route("/search", search_page),
        Route("/finding/{id:path}", finding_page),
        WebSocketRoute("/ws", ws_updates),
        Route("/bridge/messages", bridge_messages),
        Route("/bridge/agents", bridge_agents),
        Route("/bridge/watch", bridge_watch),
        Route("/bridge/send", bridge_send, methods=["POST"]),
        Route("/kb/search", kb_search),
        Route("/federation/search", federated_search, methods=["POST"]),
        Route("/kb/recent", kb_recent),
        Route("/kb/finding/{id:path}", finding_get),
        Route("/issues", issues_list),
        Route("/issues/{id:path}", issue_get),
    ]
    return Starlette(routes=routes, on_startup=[on_startup])
