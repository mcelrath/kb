"""CLI handler for the serve command."""

import sys


def run_serve(kb, args) -> None:
    try:
        import uvicorn
        SERVE_AVAILABLE = True
    except ImportError:
        SERVE_AVAILABLE = False

    if not SERVE_AVAILABLE:
        print("Error: starlette and uvicorn required for 'kb serve'")
        print("Install with: pip install starlette uvicorn")
        sys.exit(1)

    from kb.server import create_app
    from kb.server.auth import resolve_server_token
    app = create_app(kb)
    print(f"Starting KB server at http://{args.host}:{args.port}")
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        if resolve_server_token():
            print("Federation auth: bearer token configured — non-loopback federation endpoints require it.")
        else:
            print("WARNING: bound to a routable interface with NO server token. Federation endpoints "
                  "default-deny all non-loopback access; set KB_SERVER_TOKEN (or config.toml [server] token) "
                  "to enable peer federation. (Existing bridge/web endpoints remain open.)")
    print("WebSocket live updates enabled at /ws")
    # timeout_graceful_shutdown bounds SIGTERM handling: the bridge /watch SSE and
    # /ws WebSocket are infinite streams that never drain on their own, so without
    # a cap uvicorn waits forever on shutdown and `systemctl restart` hangs until
    # the unit's TimeoutStopSec (90s) SIGKILLs it. 5s force-closes lingering
    # streams (clients reconnect) so restart completes promptly.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                timeout_graceful_shutdown=5)
