"""P-sec (kb-8bb277): federated-kb server bearer-token auth — default-deny for
non-loopback, loopback always allowed, token gates correctly."""

import types

from kb.server import auth


def _req(host, authorization=None):
    headers = {}
    if authorization is not None:
        headers["authorization"] = authorization
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host) if host is not None else None,
        headers=headers,
    )


def test_loopback_always_allowed(monkeypatch):
    monkeypatch.delenv("KB_SERVER_TOKEN", raising=False)
    monkeypatch.setattr(auth, "resolve_server_token", lambda: None)
    for h in ("127.0.0.1", "::1", "localhost", None):
        assert auth.request_authorized(_req(h)) is True


def test_nonloopback_denied_without_token(monkeypatch):
    monkeypatch.setattr(auth, "resolve_server_token", lambda: None)
    # no server token configured => remote access is fail-closed denied
    assert auth.request_authorized(_req("10.0.0.5")) is False
    assert auth.request_authorized(_req("10.0.0.5", "Bearer whatever")) is False


def test_nonloopback_requires_matching_bearer(monkeypatch):
    monkeypatch.setattr(auth, "resolve_server_token", lambda: "s3cr3t")
    assert auth.request_authorized(_req("10.0.0.5")) is False
    assert auth.request_authorized(_req("10.0.0.5", "Bearer wrong")) is False
    assert auth.request_authorized(_req("10.0.0.5", "Bearer s3cr3t")) is True
    # loopback still exempt even when a token is set
    assert auth.request_authorized(_req("127.0.0.1")) is True


def test_resolve_token_prefers_env(monkeypatch):
    monkeypatch.setenv("KB_SERVER_TOKEN", "env-tok")
    assert auth.resolve_server_token() == "env-tok"
    # blank env is ignored (stripped); falls through to config.toml (or None)
    monkeypatch.setenv("KB_SERVER_TOKEN", "   ")
    assert auth.resolve_server_token() != "   "
