"""P2a (kb-4afbb8): /federation/search endpoint — auth-gated, returns local search
results with owner + model_id + raw-cosine similarity (epic kb-907fc8)."""

import asyncio
import json
import types

import kb.server.auth as auth
from kb.server.federation import make_federation_handlers, PROTOCOL_VERSION


class _FakeEmb:
    embedding_model = "qwen3-8b"
    embedding_dim = 4096


class _FakeKB:
    _embedding = _FakeEmb()

    def search(self, q, limit=10):
        return [{"id": "kb-1", "type": "discovery", "summary": "a hit",
                 "project": "kb", "similarity": 0.91, "content": "x"}]


def _req(host, body, authorization=None):
    headers = {}
    if authorization is not None:
        headers["authorization"] = authorization

    async def _json():
        return body

    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host), headers=headers, json=_json,
    )


def test_loopback_search_ok():
    h = make_federation_handlers(_FakeKB())
    resp = asyncio.run(h(_req("127.0.0.1", {"query": "foo", "k": 5})))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["model_id"] == "qwen3-8b" and body["dim"] == 4096
    r0 = body["results"][0]
    assert r0["id"] == "kb-1" and r0["similarity"] == 0.91 and r0["owner"]


def test_nonloopback_denied_without_token(monkeypatch):
    monkeypatch.setattr(auth, "resolve_server_token", lambda: None)
    h = make_federation_handlers(_FakeKB())
    resp = asyncio.run(h(_req("10.0.0.9", {"query": "foo"})))
    assert resp.status_code == 401


def test_nonloopback_allowed_with_token(monkeypatch):
    monkeypatch.setattr(auth, "resolve_server_token", lambda: "tok")
    h = make_federation_handlers(_FakeKB())
    resp = asyncio.run(h(_req("10.0.0.9", {"query": "foo"}, "Bearer tok")))
    assert resp.status_code == 200


def test_empty_query_rejected():
    h = make_federation_handlers(_FakeKB())
    resp = asyncio.run(h(_req("127.0.0.1", {"k": 5})))
    assert resp.status_code == 400
