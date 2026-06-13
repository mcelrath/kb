"""Live-update WebSocket handler and background poller.

Extracted verbatim from kb.py:1804-1845.

State (connected_clients, last_state) is factory-local: create_app() creates
a fresh instance per server process so the poller has a clean slate.
"""

import asyncio

from starlette.websockets import WebSocket


def make_live_handlers(kb):
    """Return (ws_updates, on_startup) bound to a fresh live state.

    connected_clients and last_state are closure-local so each create_app()
    invocation gets its own independent state (mirrors the original closure
    behavior inside main()).
    """
    connected_clients: set = set()
    last_state = {"count": 0, "latest": ""}

    async def ws_updates(websocket: WebSocket):
        await websocket.accept()
        connected_clients.add(websocket)
        try:
            # Send current state on connect
            count, latest = kb.get_latest_update()
            await websocket.send_json({"type": "state", "count": count, "latest": latest})
            # Keep connection alive
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await websocket.send_json({"type": "ping"})
        except Exception:
            pass
        finally:
            connected_clients.discard(websocket)

    async def check_for_updates():
        """Background task to check for DB changes and notify clients."""
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            if connected_clients:
                count, latest = kb.get_latest_update()
                if count != last_state["count"] or latest != last_state["latest"]:
                    last_state["count"] = count
                    last_state["latest"] = latest
                    # Broadcast to all connected clients
                    dead = set()
                    for ws in connected_clients:
                        try:
                            await ws.send_json({"type": "update", "count": count, "latest": latest})
                        except Exception:
                            dead.add(ws)
                    connected_clients.difference_update(dead)

    async def on_startup():
        asyncio.create_task(check_for_updates())

    return ws_updates, on_startup
