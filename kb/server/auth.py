"""Federated-kb server auth (epic kb-907fc8 P-sec).

Bearer-token gate for NON-loopback access to sensitive (federation) endpoints.

Default-deny: a non-loopback request with no configured server token — or a wrong
token — is denied. Loopback (local agents / hooks / the web UI) is always allowed.

IMPORTANT scoping: this does NOT globally gate the existing bridge/web endpoints —
the live cross-host agent bridge relies on them being reachable. Only handlers that
explicitly call `require_authorized()` enforce the token. The new federated-search
endpoints (P2a) call it, so the NEW exposure is auth-gated from its first commit
(this is what "auth lands with the bind-open" means here). A coordinated flip to
all-endpoint default-deny is P6 (full hardening), once every peer carries a token.

Token source (first non-empty wins): KB_SERVER_TOKEN env, then config.toml
[server] token. None => non-loopback federation access is denied (fail-closed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_LOOPBACK = {"127.0.0.1", "::1", "localhost", None}


def resolve_server_token() -> str | None:
    """The bearer token this server accepts for federation, or None if unset."""
    tok = os.environ.get("KB_SERVER_TOKEN", "").strip()
    if tok:
        return tok
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import,no-redef]
        except ImportError:
            return None
    p = Path.home() / ".config" / "kb" / "config.toml"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
        v = (data.get("server") or {}).get("token")
        return str(v) if v else None
    except Exception:
        return None


def _client_host(request: Any) -> str | None:
    try:
        return request.client.host if request.client else None
    except Exception:
        return None


def request_authorized(request: Any) -> bool:
    """True if `request` may access a federation endpoint.

    Loopback is always allowed (local use). A non-loopback request requires a
    configured server token AND a matching `Authorization: Bearer <token>` header;
    otherwise it is denied (default-deny / fail-closed).
    """
    host = _client_host(request)
    if host in _LOOPBACK:
        return True
    token = resolve_server_token()
    if not token:
        return False
    auth = ""
    try:
        auth = request.headers.get("authorization", "") or ""
    except Exception:
        return False
    return auth == f"Bearer {token}"
