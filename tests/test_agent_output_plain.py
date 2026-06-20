"""Regression guard (kb-431950 T7): under KB_AGENT=1, EVERY list/row command must
emit ZERO ANSI escapes and NOT truncate — even at a tiny terminal width. This is
the end-to-end backstop ensuring color/truncation never leaks into agent-parsed
output (the whole point of the AGENT_MODE gate).

Runs the real CLI entrypoints in a subprocess (the agent path), so it catches a
regression in ANY command — not just the ones with unit tests.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (entry script, argv) — one per list/row-producing command across kb + kbt.
KB = "kb.py"
KBT = "kb/issue_cli.py"
COMMANDS = [
    (KB, ["search", "the", "-n", "5"]),
    (KB, ["list", "-n", "5"]),
    (KB, ["stats"]),
    (KB, ["surface", "--query", "test", "-n", "3"]),
    (KBT, ["list"]),
    (KBT, ["ready"]),
    (KBT, ["blocked"]),
    (KBT, ["children", "kb-431950"]),
]

ESC = b"\x1b["  # CSI — the start of every ANSI SGR color escape


def _run(script, argv):
    env = dict(os.environ, KB_AGENT="1", COLUMNS="20", CLAUDECODE="1")
    return subprocess.run(
        [sys.executable, script, *argv],
        cwd=REPO, env=env, capture_output=True, timeout=120,
    )


@pytest.mark.parametrize("script,argv", COMMANDS, ids=[f"{s}:{a[0]}" for s, a in COMMANDS])
def test_agent_output_has_no_ansi(script, argv):
    """No ANSI escape may appear in agent-mode stdout, even at COLUMNS=20."""
    p = _run(script, argv)
    assert ESC not in p.stdout, (
        f"{script} {argv} leaked an ANSI escape in agent mode:\n{p.stdout[:300]!r}"
    )


@pytest.mark.parametrize("script,argv", COMMANDS, ids=[f"{s}:{a[0]}" for s, a in COMMANDS])
def test_agent_output_not_truncated(script, argv):
    """Agent mode must not truncate: no line carries our ellipsis marker added by
    fit_line/truncate. (A literal '…' inside content is fine; fit_line appends it
    only on a width cut, which must never happen for agents — term_width is None.)
    Heuristic: at COLUMNS=20, a row-producing command would be cut if truncation
    were active, so assert no line was shortened to ~20 cols ending in the marker."""
    p = _run(script, argv)
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        # fit_line cut → exactly width-1 visible chars + '…' (+ RESET, stripped
        # since there's no ANSI in agent mode). The tell is a trailing ellipsis
        # on a line near the tiny width. Agent mode must never produce that.
        assert not (len(line) <= 21 and line.endswith("…")), (
            f"{script} {argv} appears truncated in agent mode: {line!r}"
        )
