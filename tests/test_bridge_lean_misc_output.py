"""Tests for colorization and row-truncation in bridge.py, lean.py, misc.py.

Strategy:
- bridge: test _run_search (row-producing, callable with fake kb + results)
- lean:   test run_queue_defer listing (row-producing, callable with a fake kb)
- misc:   test run_reconcile summary rows (via fake reconciler via monkeypatch)

For each:
- agent mode (AGENT_MODE=True): no \033[ escapes, no truncation
- user mode  (AGENT_MODE=False, small term_width): ANSI escapes present, rows fit width
"""
import types
import pytest

from kb.cli import output
from kb.cli.commands import bridge as bridge_mod
from kb.cli.commands import lean as lean_mod
from kb.cli.commands import misc as misc_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_bridge_kb(results):
    """Minimal fake kb object whose _bridge.search returns `results`."""
    class FakeBridge:
        def search(self, query, limit=10):
            return results

    class FakeKB:
        _bridge = FakeBridge()

    return FakeKB()


def _fake_queue_kb(rows):
    """Minimal fake kb for run_queue_defer --list."""
    class FakeKB:
        def list_deferred_queue_rows(self, limit=50):
            return rows

    return FakeKB()


# ---------------------------------------------------------------------------
# bridge._run_search — agent mode
# ---------------------------------------------------------------------------

def test_bridge_search_agent_no_color(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    results = [
        {"id": 42, "sender": "alice", "similarity": 0.85,
         "subject": "hello world", "body": "some body text"},
    ]
    args = types.SimpleNamespace(query="q", limit=5)
    bridge_mod._run_search(_fake_bridge_kb(results), args)
    out = capsys.readouterr().out
    assert "\033[" not in out
    # full subject present (not truncated)
    assert "hello world" in out


def test_bridge_search_user_color_and_truncate(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 40)
    results = [
        {"id": 7, "sender": "bob", "similarity": 0.55,
         "subject": "A" * 200, "body": "B" * 200},
    ]
    args = types.SimpleNamespace(query="q", limit=5)
    bridge_mod._run_search(_fake_bridge_kb(results), args)
    out = capsys.readouterr().out
    # color escapes present
    assert "\033[" in out
    # every output line fits within 40 visible chars
    for line in out.splitlines():
        assert output.visible_len(line) <= 40, f"line too wide: {repr(line)}"


# ---------------------------------------------------------------------------
# bridge._run_recv — agent mode (needs kb-server; test the listing branch only)
# ---------------------------------------------------------------------------

def test_bridge_recv_listing_agent_no_color(monkeypatch, capsys):
    """Directly exercise the listing loop in _run_recv via monkeypatching urlopen."""
    import json
    import urllib.request

    msgs = [
        {"id": 3, "sender": "carol", "subject": "test subj", "body": ""},
    ]

    class FakeResp:
        def read(self):
            return json.dumps(msgs).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=8: FakeResp())
    monkeypatch.setattr(output, "AGENT_MODE", True)

    args = types.SimpleNamespace(agent_id="carol", limit=50, from_id=None)
    bridge_mod._run_recv(args)
    out = capsys.readouterr().out
    assert "\033[" not in out
    assert "carol" in out or "test subj" in out


def test_bridge_recv_listing_user_color(monkeypatch, capsys):
    import json
    import urllib.request

    msgs = [
        {"id": 5, "sender": "dave", "subject": "S" * 200, "body": ""},
    ]

    class FakeResp:
        def read(self):
            return json.dumps(msgs).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=8: FakeResp())
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 40)

    args = types.SimpleNamespace(agent_id="dave", limit=50, from_id=None)
    bridge_mod._run_recv(args)
    out = capsys.readouterr().out
    assert "\033[" in out
    for line in out.splitlines():
        assert output.visible_len(line) <= 40


# ---------------------------------------------------------------------------
# lean.run_queue_defer — agent mode
# ---------------------------------------------------------------------------

_SAMPLE_ROWS = [
    ("rowid-abc123", "/path/to/Foo.lean", "myDecl", "sorry", "ready",
     "data_blocked_on:", "waiting for X", "2026-01-01T00:00:00"),
]


def test_lean_queue_defer_list_agent_no_color(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    args = types.SimpleNamespace(list=True, row_id=None)
    with pytest.raises(SystemExit):
        lean_mod.run_queue_defer(_fake_queue_kb(_SAMPLE_ROWS), args)
    out = capsys.readouterr().out
    assert "\033[" not in out
    # full content present (not truncated)
    assert "rowid-abc" in out


def test_lean_queue_defer_list_user_color_and_truncate(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 50)
    args = types.SimpleNamespace(list=True, row_id=None)
    with pytest.raises(SystemExit):
        lean_mod.run_queue_defer(_fake_queue_kb(_SAMPLE_ROWS), args)
    out = capsys.readouterr().out
    assert "\033[" in out
    # row lines (indented with "  ") truncated to <=50 visible chars
    for line in out.splitlines():
        if line.startswith("  "):
            assert output.visible_len(line) <= 50, f"row too wide: {repr(line)}"


# ---------------------------------------------------------------------------
# misc.run_reconcile — delegates to the real DocumentReconciler API (B3 fix).
# The handler now constructs DocumentReconciler(kb, project), calls
# reconcile(doc_dir), and prints reconciler.format_report(report) — no in-handler
# colorization. It validates: --project required, document must be a directory.
# ---------------------------------------------------------------------------

def _patch_reconciler(monkeypatch):
    import sys

    class FakeReconciler:
        def __init__(self, kb, project):
            self.project = project

        def reconcile(self, doc_dir):
            return {"stub": True}

        def format_report(self, report):
            return "KB RECONCILIATION REPORT\nWell matched: 3"

        def export_missing_json(self, report):
            return [{"content": "x"}]

    fake_mod = types.ModuleType("kb_reconcile")
    fake_mod.DocumentReconciler = FakeReconciler
    monkeypatch.setitem(sys.modules, "kb_reconcile", fake_mod)


def test_misc_reconcile_prints_format_report(monkeypatch, capsys, tmp_path):
    _patch_reconciler(monkeypatch)
    args = types.SimpleNamespace(document=str(tmp_path), project="proj", export_missing=None)
    misc_mod.run_reconcile(None, args)
    assert "Well matched: 3" in capsys.readouterr().out


def test_misc_reconcile_requires_project(monkeypatch, tmp_path):
    _patch_reconciler(monkeypatch)
    args = types.SimpleNamespace(document=str(tmp_path), project=None, export_missing=None)
    with pytest.raises(SystemExit):
        misc_mod.run_reconcile(None, args)


def test_misc_reconcile_requires_directory(monkeypatch, tmp_path):
    _patch_reconciler(monkeypatch)
    f = tmp_path / "doc.md"
    f.write_text("x")
    args = types.SimpleNamespace(document=str(f), project="proj", export_missing=None)
    with pytest.raises(SystemExit):
        misc_mod.run_reconcile(None, args)


def test_misc_reconcile_export_missing(monkeypatch, tmp_path):
    import json
    _patch_reconciler(monkeypatch)
    outp = tmp_path / "missing.json"
    args = types.SimpleNamespace(document=str(tmp_path), project="proj", export_missing=str(outp))
    misc_mod.run_reconcile(None, args)
    assert json.loads(outp.read_text()) == [{"content": "x"}]
