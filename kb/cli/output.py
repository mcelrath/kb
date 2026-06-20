"""Shared CLI output helpers: agent/user mode, color, terminal width, and
ANSI-aware truncation. Importable by kb.py, kb/issue_cli.py (kbt), and every
kb/cli/commands/* module so the output discipline is consistent:

  - colorized for users, PLAIN for agents (gated by KB_AGENT / CLAUDECODE / TTY);
  - one-line ROW entries truncated to the terminal width for users — NEVER for
    agents/pipes, and never for multi-line / full-content views.

Mode is resolved once at import (`AGENT_MODE`). Tests monkeypatch that attribute;
the helpers read it by name at call time so the patch takes effect.
"""
from __future__ import annotations

import os
import re
import shutil
import sys


def is_agent() -> bool:
    """True when output is consumed by an agent (or a pipe), not a human terminal.

    KB_AGENT=1/0 is an explicit override; CLAUDECODE=1 is set by Claude Code in
    all subprocesses; otherwise a non-TTY stdout (pipe/subprocess) = agent.
    Mirrors kb.py::_is_agent_context so the two never diverge.
    """
    override = os.environ.get("KB_AGENT", "")
    if override == "1":
        return True
    if override == "0":
        return False
    if os.environ.get("CLAUDECODE") == "1":
        return True
    return not sys.stdout.isatty()


AGENT_MODE = is_agent()

# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------
RESET = "\033[0m"
_NAMED = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "dim": "\033[2m", "bold": "\033[1m",
}


def c(text: str, color: str | None) -> str:
    """Wrap `text` in `color` (a name in _NAMED or a raw SGR code). No-op in
    agent mode or when `color` is falsy."""
    if AGENT_MODE or not color:
        return text
    code = _NAMED.get(color, color if color.startswith("\033") else "")
    return f"{code}{text}{RESET}" if code else text


def sim_color(sim: float) -> str:
    """Threshold color for a cosine similarity (green/yellow/red)."""
    return "green" if sim >= 0.7 else "yellow" if sim >= 0.5 else "red"


# ---------------------------------------------------------------------------
# Terminal width + ANSI-aware truncation
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(s: str) -> int:
    """Length of `s` ignoring ANSI SGR escapes (what the terminal renders)."""
    return len(_ANSI_RE.sub("", s))


def term_width(default: int = 100) -> int | None:
    """Terminal column count for users; None for agents/pipes (= do not truncate)."""
    if AGENT_MODE:
        return None
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
        return cols if cols and cols > 0 else default
    except OSError:
        return default


def truncate(text: str, width: int | None) -> str:
    """Truncate `text` to <= `width` VISIBLE chars (ANSI escapes don't count),
    appending an ellipsis when cut and a RESET so color never bleeds. A falsy or
    non-positive `width`, or text already within width, is returned unchanged."""
    if not width or width <= 0 or visible_len(text) <= width:
        return text
    out: list[str] = []
    vis = 0
    i = 0
    n = len(text)
    limit = max(1, width - 1)  # reserve one column for the ellipsis
    while i < n and vis < limit:
        m = _ANSI_RE.match(text, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(text[i])
        vis += 1
        i += 1
    return "".join(out) + "…" + ("" if AGENT_MODE else RESET)


def fit_line(line: str) -> str:
    """Truncate a single ROW to the terminal width in user mode; passthrough for
    agents/pipes. Use ONLY for one-line list/row entries, never full views."""
    return truncate(line, term_width())
