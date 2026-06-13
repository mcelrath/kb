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
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
