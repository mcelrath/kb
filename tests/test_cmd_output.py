"""Tests for colorization and truncation in admin.py and maintenance.py handlers."""
import types
import pytest

from kb.cli import output
from kb.cli.commands import admin, maintenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_kb_stats(db_path="/tmp/test.db"):
    """Return a stats dict matching what kb.stats() produces."""
    return {
        "db_path": db_path,
        "total": 42,
        "current": 40,
        "superseded": 2,
        "no_summary": 0,
        "no_embedding": 0,
        "no_summary_by_project": {},
        "no_embedding_by_project": {},
        "by_type": {"discovery": 30, "insight": 12},
        "by_project": {"project-a": 25, "project-b": 17},
    }


def _fake_kb_review():
    """Return a review_queue result with one category and two items."""
    return {
        "stale": [
            {"id": "kb-20240101-aabbcc", "project": "proj-x",
             "content": "A" * 80},
            {"id": "kb-20240102-ddeeff", "project": None,
             "content": "B" * 80},
        ]
    }


class _FakeKB:
    def stats(self):
        return _fake_kb_stats()

    def review_queue(self, project=None, limit=20):
        return _fake_kb_review()


def _make_stats_args():
    return types.SimpleNamespace()


def _make_review_args():
    return types.SimpleNamespace(project=None, limit=20)


# ---------------------------------------------------------------------------
# run_stats — agent mode
# ---------------------------------------------------------------------------

def test_run_stats_agent_no_ansi(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    admin.run_stats(_FakeKB(), _make_stats_args())
    captured = capsys.readouterr().out
    assert "\033[" not in captured


def test_run_stats_agent_not_truncated(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    admin.run_stats(_FakeKB(), _make_stats_args())
    captured = capsys.readouterr().out
    # All expected content present (db path, counts, type/project names)
    assert "/tmp/test.db" in captured
    assert "42" in captured
    assert "discovery" in captured
    assert "project-a" in captured


# ---------------------------------------------------------------------------
# run_stats — user mode
# ---------------------------------------------------------------------------

def test_run_stats_user_colorized(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 120)
    admin.run_stats(_FakeKB(), _make_stats_args())
    captured = capsys.readouterr().out
    assert "\033[" in captured


def test_run_stats_user_has_content(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 120)
    admin.run_stats(_FakeKB(), _make_stats_args())
    captured = capsys.readouterr().out
    assert "42" in captured
    assert "discovery" in captured
    assert "project-a" in captured


# ---------------------------------------------------------------------------
# run_review — agent mode
# ---------------------------------------------------------------------------

def test_run_review_agent_no_ansi(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    maintenance.run_review(_FakeKB(), _make_review_args())
    captured = capsys.readouterr().out
    assert "\033[" not in captured


def test_run_review_agent_not_truncated(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    maintenance.run_review(_FakeKB(), _make_review_args())
    captured = capsys.readouterr().out
    # Both items should appear
    assert "kb-20240101-aabbcc" in captured
    assert "kb-20240102-ddeeff" in captured


# ---------------------------------------------------------------------------
# run_review — user mode with small terminal
# ---------------------------------------------------------------------------

def test_run_review_user_colorized(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 60)
    maintenance.run_review(_FakeKB(), _make_review_args())
    captured = capsys.readouterr().out
    assert "\033[" in captured


def test_run_review_user_rows_truncated(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    width = 50
    monkeypatch.setattr(output, "term_width", lambda default=100: width)
    maintenance.run_review(_FakeKB(), _make_review_args())
    captured = capsys.readouterr().out
    # Each data row (starting with spaces) must fit within the terminal width
    for line in captured.splitlines():
        if line.startswith("  kb-"):
            assert output.visible_len(line) <= width, (
                f"row too wide ({output.visible_len(line)} > {width}): {line!r}"
            )
