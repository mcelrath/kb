"""P3 (kb-536da0): `kb search --federated` client — fan out to peers, merge by raw
cosine similarity, tolerate offline peers (epic kb-907fc8)."""

import json
import types

import kb.cli.commands.fed_search as fed


class _FakeEmb:
    embedding_model = "qwen3-8b"


class _Peers:
    def list(self, enabled_only=False):
        return [{"url": "http://p1:8765", "token": "t", "label": "p1"}]


class _FakeKB:
    _embedding = _FakeEmb()
    _peers = _Peers()

    def search(self, q, limit=10):
        return [{"id": "kb-local", "summary": "local hit", "project": "kb",
                 "similarity": 0.80, "type": "discovery"}]


def _args(json_out=False):
    return types.SimpleNamespace(query="foo", limit=10, json=json_out)


def test_peer_outranks_local_and_both_present(monkeypatch, capsys):
    def fake_query(peer, query, k):
        return {"owner": "p1", "model_id": "qwen3-8b", "results": [
            {"id": "kb-peer", "summary": "peer hit", "similarity": 0.95,
             "owner": "p1", "model_id": "qwen3-8b"}]}
    monkeypatch.setattr(fed, "_query_peer", fake_query)
    fed.run_federated_search(_FakeKB(), _args())
    out = capsys.readouterr().out
    assert "kb-peer" in out and "kb-local" in out
    assert out.index("kb-peer") < out.index("kb-local")  # 0.95 ranks above 0.80
    assert "[p1]" in out and "[(local)]" in out


def test_offline_peer_tolerated(monkeypatch, capsys):
    def boom(peer, query, k):
        raise OSError("connection refused")
    monkeypatch.setattr(fed, "_query_peer", boom)
    fed.run_federated_search(_FakeKB(), _args())
    out = capsys.readouterr().out
    assert "kb-local" in out          # local results still shown
    assert "unreachable" in out       # offline peer reported, not fatal


def test_json_output(monkeypatch, capsys):
    monkeypatch.setattr(fed, "_query_peer", lambda p, q, k: {"results": []})
    fed.run_federated_search(_FakeKB(), _args(json_out=True))
    d = json.loads(capsys.readouterr().out)
    assert d["results"][0]["id"] == "kb-local"
