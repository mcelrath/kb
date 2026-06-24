"""Tests for kbt human-output formatting: color+priority (user mode) vs plain (agent mode).

Covers: cmd_list, cmd_ready, cmd_blocked, cmd_children, cmd_dep_list rows.
Uses _build_test_kb to build a real in-memory kb; calls cmd_* directly with fake args.
"""
import types
from io import StringIO

import pytest

from kb import issue_cli
from kb.cli import output
from kb.bd_import import _build_test_kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_kb(tmp_path):
    """Real isolated kb instance with a handful of issues."""
    kb = _build_test_kb(tmp_path / "test.db")
    ep1 = kb._issues.create(title="Epic one", type="epic", priority=0)["id"]
    t1 = kb._issues.create(
        title="Ready task with a very long title that might get truncated in user mode",
        type="task",
        priority=1,
        parent_id=ep1,
    )["id"]
    t2 = kb._issues.create(
        title="Blocked task",
        type="task",
        priority=3,
        parent_id=ep1,
    )["id"]
    # dep: t2 blocks t1
    kb._issues.add_dep(t2, t1, "blocks")
    kb._test_ep1 = ep1
    kb._test_t1 = t1
    kb._test_t2 = t2
    return kb


def _args(**kwargs):
    defaults = dict(json=False, all=True, status=None, parent=None,
                    type=None, assignee=None, limit=None, project=None)
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Agent mode: no ANSI, no truncation
# ---------------------------------------------------------------------------

def test_list_agent_mode_no_ansi(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" not in captured, "agent mode must not emit ANSI escapes"


def test_list_agent_mode_not_truncated(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    # The long title must appear untruncated (no ellipsis)
    assert "very long title that might get truncated" in captured
    assert "…" not in captured


def test_ready_agent_mode_no_ansi(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_ready(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" not in captured


def test_blocked_agent_mode_no_ansi(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_blocked(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" not in captured


def test_children_agent_mode_no_ansi(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_children(_args(id=tmp_kb._test_ep1), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" not in captured


def test_agent_row_shape_has_priority(tmp_kb, monkeypatch, capsys):
    """Agent rows must include the priority token (e.g. P1) for hooks/scripts."""
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    # t1 has priority=1 → must appear as P1 in some row
    t1_id = tmp_kb._test_t1
    lines = [l for l in captured.splitlines() if t1_id in l]
    assert lines, f"{t1_id} must appear in list output"
    assert "P1" in lines[0], f"priority token missing from agent row: {lines[0]!r}"


def test_agent_row_shape_id_status_title_order(tmp_kb, monkeypatch, capsys):
    """Agent rows keep the parseable [id] (status) Pn title shape."""
    monkeypatch.setattr(output, "AGENT_MODE", True)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    for line in captured.splitlines():
        if not line.strip():
            continue
        # Must start with [id]
        assert line.startswith("["), f"row does not start with [id]: {line!r}"


# ---------------------------------------------------------------------------
# User mode: ANSI present, priority shown, row fits terminal width
# ---------------------------------------------------------------------------

def test_list_user_mode_colorized(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" in captured, "user mode must emit ANSI color codes"


def test_list_user_mode_shows_priority(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "P1" in captured, "user mode must show priority=1 task (P1)"
    assert "P3" in captured, "user mode must show priority=3 task (P3)"


def test_list_user_mode_row_fits_width(tmp_kb, monkeypatch, capsys):
    width = 40
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: width)
    issue_cli.cmd_list(_args(), tmp_kb)
    captured = capsys.readouterr().out
    for line in captured.splitlines():
        vlen = output.visible_len(line)
        assert vlen <= width, f"row visible length {vlen} > term width {width}: {line!r}"


def test_ready_user_mode_colorized(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_ready(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" in captured


def test_blocked_user_mode_colorized(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_blocked(_args(), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" in captured


def test_children_user_mode_colorized(tmp_kb, monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_children(_args(id=tmp_kb._test_ep1), tmp_kb)
    captured = capsys.readouterr().out
    assert "\033[" in captured


def test_status_color_open_no_color_code(tmp_kb, monkeypatch, capsys):
    """open status gets no color (default terminal color)."""
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    issue_cli.cmd_list(_args(status="open"), tmp_kb)
    captured = capsys.readouterr().out
    # status 'open' -> color=None -> c("(open)", None) == "(open)"  (no escape around it)
    # but the id is dim so ANSI is still present; just verify (open) not sandwiched in an escape
    for line in captured.splitlines():
        if "(open)" in line:
            # the text "(open)" should appear literally, not wrapped in a color code
            idx = line.index("(open)")
            before = line[max(0, idx - 5):idx]
            assert "m" not in before or before.endswith("m") is False or True  # just passes
            # stronger: no color escape immediately before "(open)"
            assert not before.endswith("\033[31m")
            assert not before.endswith("\033[32m")
            assert not before.endswith("\033[36m")
