"""Unit tests for the /bridge/messages endpoint.

Uses Starlette's TestClient with a mock kb object and patched
BRIDGE_MESSAGES_PATH so no live server or filesystem state is required.

Covers:
1. GET /bridge/messages?since=50       -> only messages with id > 50
2. GET /bridge/messages?recipient=bob  -> only messages to bob or 'all'
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from kb.server.api import make_api_handlers
from kb.server.bridge import bridge_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb():
    """Return a mock kb that records calls and returns canned data."""
    kb = MagicMock()
    kb.list_findings.return_value = []
    kb.search.return_value = []
    kb._issues = MagicMock()
    kb._issues.list.return_value = []
    kb._issues.get.return_value = None
    return kb


def _make_messages_file(messages: list[dict]) -> Path:
    """Write messages as jsonl to a temp file and return the Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for m in messages:
        tmp.write(json.dumps(m) + "\n")
    tmp.close()
    return Path(tmp.name)


def _build_app(kb):
    """Build a minimal Starlette app with only the /bridge/messages route."""
    make_api_handlers(kb)  # exercise the factory (no /moim route any more)
    routes = [Route("/bridge/messages", bridge_messages)]
    return Starlette(routes=routes)


# ---------------------------------------------------------------------------
# Test 1: GET /bridge/messages?since=50 returns only messages with id > 50
# ---------------------------------------------------------------------------

class TestBridgeMessagesSince:
    def _make_msgs_path(self):
        messages = [
            {"id": 49, "sender": "a", "to": ["all"], "subject": "before", "body": "before body", "ts": "t1"},
            {"id": 50, "sender": "a", "to": ["all"], "subject": "at cursor", "body": "at body", "ts": "t2"},
            {"id": 51, "sender": "a", "to": ["all"], "subject": "after", "body": "after body", "ts": "t3"},
            {"id": 75, "sender": "b", "to": ["all"], "subject": "well after", "body": "well after body", "ts": "t4"},
        ]
        return _make_messages_file(messages)

    def test_since_excludes_at_and_below(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?since=50")
            assert r.status_code == 200
            ids = [m["id"] for m in r.json()]
            assert 49 not in ids
            assert 50 not in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_since_includes_above(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?since=50")
            ids = [m["id"] for m in r.json()]
            assert 51 in ids
            assert 75 in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_returns_json_list(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?since=50")
            assert r.status_code == 200
            assert isinstance(r.json(), list)
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_since_zero_includes_all_above(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?since=0")
            ids = [m["id"] for m in r.json()]
            assert set(ids) == {49, 50, 51, 75}
        finally:
            msgs_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 2: GET /bridge/messages?recipient=bob filters by recipient
# ---------------------------------------------------------------------------

class TestBridgeMessagesRecipient:
    def _make_msgs_path(self):
        messages = [
            {"id": 1, "sender": "x", "to": ["bob"], "subject": "for bob", "body": "bob body", "ts": "t1"},
            {"id": 2, "sender": "x", "to": ["alice"], "subject": "for alice", "body": "alice body", "ts": "t2"},
            {"id": 3, "sender": "x", "to": ["all"], "subject": "broadcast", "body": "broadcast body", "ts": "t3"},
            {"id": 4, "sender": "x", "to": ["bob", "carol"], "subject": "multi", "body": "multi body", "ts": "t4"},
        ]
        return _make_messages_file(messages)

    def test_bob_gets_his_messages(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?recipient=bob")
            assert r.status_code == 200
            ids = [m["id"] for m in r.json()]
            assert 1 in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_bob_gets_broadcasts(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?recipient=bob")
            ids = [m["id"] for m in r.json()]
            assert 3 in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_bob_gets_multi_recipient_message(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?recipient=bob")
            ids = [m["id"] for m in r.json()]
            assert 4 in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_bob_does_not_get_alice_messages(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?recipient=bob")
            ids = [m["id"] for m in r.json()]
            assert 2 not in ids
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_alice_gets_her_messages(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/bridge/messages?recipient=alice")
            ids = [m["id"] for m in r.json()]
            assert 2 in ids
            assert 1 not in ids
        finally:
            msgs_path.unlink(missing_ok=True)
