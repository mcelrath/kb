"""Tests for kb.py formatting helpers: format_results, format_finding.

Verifies:
  (a) agent-mode format_results produces NO ANSI escapes and is NOT truncated
      even with a tiny simulated terminal width;
  (b) user-mode format_results is colorized AND each row's visible length <=
      the monkeypatched term_width;
  (c) format_finding (full view) is never truncated regardless of mode.
"""
import re
import pytest

import importlib.util
import sys
import os

from kb.cli import output

# kb.py is the CLI script at the repo root; `import kb` resolves to the kb/ package,
# so load the script directly by file path.
_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb.py")
_spec = importlib.util.spec_from_file_location("_kb_script", _script_path)
kb_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kb_mod)

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

_FINDING = {
    "id": "kb-20240101-120000-abcd",
    "type": "discovery",
    "content": "A" * 200,
    "summary": "Short summary " + "x" * 60,
    "project": "knowledge-base",
    "tags": ["refactor"],
    "created_at": "2024-01-01T12:00:00Z",
    "similarity": 0.85,
}

_SECTION_FINDING = {
    "id": "kb-20240101-130000-efgh",
    "result_type": "section",
    "kind": "prose",
    "path": "/some/doc.md",
    "project": "knowledge-base",
    "content": "B" * 200,
    "heading": "Introduction",
    "similarity": 0.72,
}


def _has_ansi(text: str) -> bool:
    return bool(_ANSI_RE.search(text))


# ---------------------------------------------------------------------------
# Agent mode: no ANSI, no truncation
# ---------------------------------------------------------------------------

def test_format_results_agent_no_ansi(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    result = kb_mod.format_results([_FINDING])
    assert not _has_ansi(result), "Agent mode must produce no ANSI escapes"


def test_format_results_agent_not_truncated(monkeypatch):
    """Even if term_width were somehow non-None, agent output must not be truncated."""
    monkeypatch.setattr(output, "AGENT_MODE", True)
    monkeypatch.setattr(output, "term_width", lambda default=100: 10)  # tiny width
    result = kb_mod.format_results([_FINDING])
    # The summary field alone is 74+ chars — row must NOT be cut to 10
    assert len(result) > 10, "Agent rows must not be truncated"


def test_format_results_agent_section_no_ansi(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    result = kb_mod.format_results([_SECTION_FINDING])
    assert not _has_ansi(result)


# ---------------------------------------------------------------------------
# User mode: colorized, truncated to term_width
# ---------------------------------------------------------------------------

def test_format_results_user_colorized(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    result = kb_mod.format_results([_FINDING])
    assert _has_ansi(result), "User mode must produce ANSI color codes"


def test_format_results_user_truncated_to_width(monkeypatch):
    width = 60
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: width)
    result = kb_mod.format_results([_FINDING])
    for row in result.splitlines():
        vlen = output.visible_len(row)
        assert vlen <= width, f"Row visible length {vlen} exceeds term_width {width}: {repr(row)}"


def test_format_results_user_multiple_rows_truncated(monkeypatch):
    width = 50
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: width)
    findings = [_FINDING, _SECTION_FINDING]
    result = kb_mod.format_results(findings)
    for row in result.splitlines():
        assert output.visible_len(row) <= width


# ---------------------------------------------------------------------------
# format_finding (full/--long view) must NEVER be truncated
# ---------------------------------------------------------------------------

def test_format_finding_not_truncated_user(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 20)
    result = kb_mod.format_finding(_FINDING, verbose=True)
    # content is 200 chars — must appear untruncated in some line
    assert "A" * 50 in result, "format_finding must not truncate content"


def test_format_finding_not_truncated_agent(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    result = kb_mod.format_finding(_FINDING, verbose=True)
    assert "A" * 50 in result
    assert not _has_ansi(result), "Agent format_finding must produce no ANSI"


def test_format_finding_user_colorized(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    result = kb_mod.format_finding(_FINDING)
    assert _has_ansi(result), "User format_finding must produce ANSI colors"
