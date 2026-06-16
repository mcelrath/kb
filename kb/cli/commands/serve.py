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
    app = create_app(kb)
    print(f"Starting KB server at http://{args.host}:{args.port}")
    print("WebSocket live updates enabled at /ws")
    # timeout_graceful_shutdown bounds SIGTERM handling: the bridge /watch SSE and
    # /ws WebSocket are infinite streams that never drain on their own, so without
    # a cap uvicorn waits forever on shutdown and `systemctl restart` hangs until
    # the unit's TimeoutStopSec (90s) SIGKILLs it. 5s force-closes lingering
    # streams (clients reconnect) so restart completes promptly.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                timeout_graceful_shutdown=5)
