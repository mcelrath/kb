"""Unit tests for /moim and /bridge/messages endpoints.

Uses Starlette's TestClient with a mock kb object and patched
BRIDGE_MESSAGES_PATH so no live server or filesystem state is required.

Covers:
1. GET /moim            -> text/plain with kb findings (no bridge messages)
2. GET /moim?query=foo  -> triggers kb.search path, not kb.list_findings
3. GET /moim?recipient=alice&since=100 -> bridge filter (id > 100, to alice)
4. GET /bridge/messages?since=50       -> only messages with id > 50
5. GET /bridge/messages?recipient=bob  -> only messages to bob or 'all'
6. GET /moim with no bridge messages and no findings -> 200, empty plain text
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from kb.server.api import make_api_handlers
from kb.server.bridge import bridge_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb(findings=None, search_results=None):
    """Return a mock kb that records calls and returns canned data."""
    kb = MagicMock()
    kb.list_findings.return_value = findings if findings is not None else []
    kb.search.return_value = search_results if search_results is not None else []
    kb._issues = MagicMock()
    kb._issues.list.return_value = []
    kb._issues.get.return_value = None
    return kb


def _make_messages_file(messages: list[dict]) -> Path:
    """Write messages as jsonl to a temp file and return the Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    for m in messages:
        tmp.write(json.dumps(m) + "\n")
    tmp.close()
    return Path(tmp.name)


def _build_app(kb):
    """Build a minimal Starlette app with only the /moim and /bridge/messages routes."""
    _kb_search, _kb_recent, _finding_get, _issues_list, _issue_get, moim = make_api_handlers(kb)
    routes = [
        Route("/moim", moim),
        Route("/bridge/messages", bridge_messages),
    ]
    return Starlette(routes=routes)


def _client_with_messages(kb, messages_path: Path | None = None):
    """Build a TestClient with BRIDGE_MESSAGES_PATH optionally patched."""
    no_bridge = Path("/tmp/__no_such_bridge_messages__.jsonl")
    target = messages_path if messages_path is not None else no_bridge
    with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", target):
        app = _build_app(kb)
        client = TestClient(app, raise_server_exceptions=True)
        # Return client and the patched path so tests can make requests inside the patch
        return client, target


# ---------------------------------------------------------------------------
# Test 1: GET /moim returns text/plain with kb findings when no bridge msgs
# ---------------------------------------------------------------------------

class TestMoimPlainTextFindings:
    def test_status_200(self):
        findings = [{"id": "kb-abc", "project": "claude", "summary": "Test summary", "content": "Test content"}]
        kb = _make_kb(findings=findings)
        no_bridge = Path("/tmp/__no_bridge__.jsonl")
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", no_bridge):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert r.status_code == 200

    def test_content_type_plain_text(self):
        findings = [{"id": "kb-abc", "project": "claude", "summary": "Test summary", "content": "Test content"}]
        kb = _make_kb(findings=findings)
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert "text/plain" in r.headers.get("content-type", "")

    def test_finding_present_in_body(self):
        findings = [{"id": "kb-abc", "project": "claude", "summary": "Test summary", "content": "Test content"}]
        kb = _make_kb(findings=findings)
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert "Test summary" in r.text
        assert "Test content" in r.text

    def test_no_bridge_section_when_no_messages(self):
        findings = [{"id": "kb-abc", "project": "claude", "summary": "Test summary", "content": "Test content"}]
        kb = _make_kb(findings=findings)
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert "Unread peer messages" not in r.text

    def test_kb_section_header_present(self):
        findings = [{"id": "kb-abc", "project": "claude", "summary": "Test summary", "content": "Test content"}]
        kb = _make_kb(findings=findings)
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert "Relevant knowledge base findings" in r.text


# ---------------------------------------------------------------------------
# Test 2: GET /moim?query=foo triggers search, not list_findings
# ---------------------------------------------------------------------------

class TestMoimQuerySearch:
    def test_search_called_not_list_findings(self):
        search_results = [{"id": "kb-def", "project": "claude", "summary": "Search result", "content": "Found by search"}]
        kb = _make_kb(
            findings=[{"id": "kb-rec", "summary": "recent", "content": "recent content"}],
            search_results=search_results,
        )
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                client.get("/moim?query=foo")
        kb.search.assert_called_once()
        assert kb.search.call_args[0][0] == "foo"
        kb.list_findings.assert_not_called()

    def test_search_result_in_body(self):
        search_results = [{"id": "kb-def", "project": "claude", "summary": "Search result", "content": "Found by search"}]
        kb = _make_kb(search_results=search_results)
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim?query=foo")
        assert "Search result" in r.text

    def test_status_200(self):
        kb = _make_kb(search_results=[])
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim?query=foo")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test 3: GET /moim?recipient=alice&since=100 filters bridge messages
# ---------------------------------------------------------------------------

class TestMoimBridgeFilter:
    def _make_msgs_path(self):
        messages = [
            {"id": 99,  "sender": "bob", "to": ["alice"], "subject": "old msg", "body": "old body", "ts": "t0"},
            {"id": 100, "sender": "bob", "to": ["alice"], "subject": "at cursor", "body": "at cursor body", "ts": "t1"},
            {"id": 101, "sender": "bob", "to": ["alice"], "subject": "new msg", "body": "new body", "ts": "t2"},
            {"id": 102, "sender": "carol", "to": ["alice"], "subject": "newer msg", "body": "newer body", "ts": "t3"},
            {"id": 103, "sender": "dave", "to": ["mallory"], "subject": "wrong recipient", "body": "should not appear", "ts": "t4"},
        ]
        return _make_messages_file(messages)

    def test_since_filters_old_messages(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/moim?recipient=alice&since=100")
            assert "old msg" not in r.text
            assert "at cursor" not in r.text
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_since_includes_newer_messages(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/moim?recipient=alice&since=100")
            assert "new msg" in r.text
            assert "newer msg" in r.text
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_recipient_filter_excludes_other(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/moim?recipient=alice&since=100")
            assert "wrong recipient" not in r.text
            assert "should not appear" not in r.text
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_bridge_section_header_present(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/moim?recipient=alice&since=100")
            assert "Unread peer messages via agent-bridge" in r.text
        finally:
            msgs_path.unlink(missing_ok=True)

    def test_status_200(self):
        msgs_path = self._make_msgs_path()
        try:
            kb = _make_kb()
            with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", msgs_path):
                app = _build_app(kb)
                with TestClient(app) as client:
                    r = client.get("/moim?recipient=alice&since=100")
            assert r.status_code == 200
        finally:
            msgs_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 4: GET /bridge/messages?since=50 returns only messages with id > 50
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
            # all ids 49,50,51,75 are > 0
            assert set(ids) == {49, 50, 51, 75}
        finally:
            msgs_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 5: GET /bridge/messages?recipient=bob filters by recipient
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


# ---------------------------------------------------------------------------
# Test 6: GET /moim with no messages and no findings returns 200 empty text
# ---------------------------------------------------------------------------

class TestMoimEmpty:
    def test_status_200_not_500(self):
        kb = _make_kb(findings=[], search_results=[])
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert r.status_code == 200

    def test_body_is_empty_string(self):
        kb = _make_kb(findings=[], search_results=[])
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert r.text == ""

    def test_content_type_plain_text(self):
        kb = _make_kb(findings=[], search_results=[])
        with patch("kb.server.bridge.BRIDGE_MESSAGES_PATH", Path("/tmp/__no_bridge__.jsonl")):
            app = _build_app(kb)
            with TestClient(app) as client:
                r = client.get("/moim")
        assert "text/plain" in r.headers.get("content-type", "")
