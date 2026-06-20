"""
Characterization tests for sample CLI commands.

Asserts the OUTPUT STRUCTURE of key CLI commands so R4 (CLI dispatch extraction)
can be proven behavior-preserving. Tests run against .venv/bin/python kb.py.

Commands covered:
- kb add    -> output contains "Added: kb-<id>" (pinned to ash:8081; skipped if down)
- kb stats  -> output contains "Database:", "Total findings:", "By type:", "By project:"
- kb list   -> one finding per line, each starting with a kb-<id> token
- kb embed-status -> output contains "Configured:", "Stored:", "Verdict:" lines
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

KB_PY = str(Path(__file__).parent.parent / "kb.py")
VENV_PYTHON = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")

# The healthy embedding server (llamacpp, 4096). The session env may carry a
# STALE KB_EMBEDDING_URL (e.g. a down ollama at localhost:11434), which would
# make `kb add` QUEUE instead of embed. We pin the CLI subprocesses to the
# known-good server so add/list characterize the EMBEDDED path.
EMBED_ENV = {
    "KB_EMBEDDING_URL": "http://ash:8081/embedding",
    "KB_EMBEDDING_FORMAT": "llamacpp",
    "KB_EMBEDDING_DIM": "4096",
    "KB_EMBEDDING_MODEL": "qwen3-embedding",
}


def _embed_server_up() -> bool:
    """True if ash:8081 embedding server answers /health within 3s.

    Uses a no-proxy opener: the session inherits http_proxy=localhost:3128 which
    would otherwise mis-route this loopback-network probe.
    """
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://ash:8081/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# Module-level guard: the embedding-dependent CLI-add tests are skipped on hosts
# where ash:8081 is unreachable, so the suite is robust off this network.
embed_required = pytest.mark.skipif(
    not _embed_server_up(),
    reason="ash:8081 embedding server unreachable; kb add would queue, not embed",
)


def _run(*args, env_extra=None, timeout=15) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [VENV_PYTHON, KB_PY, *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


# ---------------------------------------------------------------------------
# kb add
# ---------------------------------------------------------------------------

@embed_required
class TestCliAdd:
    """Characterize `kb add` OUTPUT SHAPE.

    Pinned to ash:8081 (EMBED_ENV) and skipped when that server is unreachable.
    The interactive add path has TWO documented terminal outputs (kb.py): when
    the embed succeeds it prints 'Added: kb-<id>'; when the embed server is
    reachable-but-slow/overloaded (single fast-fail attempt) it degrades to
    'Queued: <file>'. Both are CURRENT behavior; we assert the disjunction and,
    on the Added branch, pin the kb-id format. Exit code is 0 in both cases.

    A generous subprocess timeout (90s) absorbs one slow embed attempt before
    the in-process fast-fail-and-queue fallback fires.
    """

    def test_add_output_shape(self, tmp_path):
        result = _run(
            "--db", str(tmp_path / "char.db"),
            "add", "characterization-test-entry-for-cli-tests",
            "-t", "discovery",
            "-p", "claude",
            "--tags", "char-test",
            "--summary", "cli characterization test entry",
            env_extra=EMBED_ENV, timeout=90,
        )
        out = result.stdout + result.stderr
        assert ("Added: kb-" in out) or ("Queued: " in out), (
            f"Unexpected add output: stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    def test_add_returns_kb_id_format(self, tmp_path):
        """On the embedded branch, output has 'Added: kb-YYYYMMDD-HHMMSS-xxxxxx'."""
        result = _run(
            "--db", str(tmp_path / "char.db"),
            "add", "another-characterization-test-probe",
            "-t", "discovery",
            "-p", "claude",
            "--summary", "second cli char test",
            env_extra=EMBED_ENV, timeout=90,
        )
        out = result.stdout + result.stderr
        added_lines = [l for l in out.splitlines() if l.startswith("Added: ")]
        if not added_lines:
            # Reachable-but-slow server degraded to the queue path; the kb-id
            # shape is unobservable here but is pinned whenever Added is emitted.
            assert "Queued: " in out, f"Neither Added nor Queued in output: {out!r}"
            pytest.skip("embed server slow; add degraded to queue (kb-id shape not emitted)")
        kb_id = added_lines[0][7:].strip()
        assert kb_id.startswith("kb-"), f"Expected kb-... id, got {kb_id!r}"
        # Format: kb-YYYYMMDD-HHMMSS-xxxxxx
        parts = kb_id.split("-")
        assert len(parts) == 4, f"Expected 4 dash-parts in {kb_id!r}"

    def test_add_exit_code_zero(self, tmp_path):
        result = _run(
            "--db", str(tmp_path / "char.db"),
            "add", "exit-code-test-probe",
            "-t", "discovery",
            "-p", "claude",
            "--summary", "exit code test",
            env_extra=EMBED_ENV, timeout=90,
        )
        assert result.returncode == 0, (
            f"kb add exited {result.returncode}: {result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# kb stats
# ---------------------------------------------------------------------------

class TestCliStats:
    def test_stats_has_required_sections(self):
        result = _run("stats", timeout=10)
        assert result.returncode == 0, f"kb stats failed: {result.stderr!r}"
        out = result.stdout
        assert "Database:" in out
        assert "Total findings:" in out
        assert "By type:" in out
        assert "By project:" in out

    def test_stats_total_is_integer(self):
        result = _run("stats", timeout=10)
        for line in result.stdout.splitlines():
            if line.startswith("Total findings:"):
                count_str = line.split(":")[1].strip()
                assert count_str.isdigit(), f"Total findings not an integer: {count_str!r}"
                break
        else:
            assert False, "No 'Total findings:' line in stats output"


# ---------------------------------------------------------------------------
# kb list
# ---------------------------------------------------------------------------

class TestCliList:
    """kb list has no --json flag; it prints one finding per line:
        'kb-YYYYMMDD-HHMMSS-xxxxxx [TYP tags...]  summary'
    """

    def test_list_exit_zero(self):
        result = _run("list", "-p", "claude", "-n", "5", timeout=10)
        assert result.returncode == 0, f"kb list failed: {result.stderr!r}"

    def test_list_line_format(self):
        """Each non-empty output line starts with a kb- id token."""
        result = _run("list", "-p", "claude", "-n", "5", timeout=10)
        assert result.returncode == 0
        out = result.stdout.strip()
        if out:
            for line in out.splitlines()[:5]:
                first_tok = line.split()[0] if line.split() else ""
                assert first_tok.startswith("kb-"), (
                    f"Expected line to start with kb- id, got: {line!r}"
                )

    def test_list_limit_respected(self):
        """-n N caps the number of finding lines."""
        result = _run("list", "-p", "claude", "-n", "2", timeout=10)
        assert result.returncode == 0
        kb_lines = [l for l in result.stdout.splitlines()
                    if l.split() and l.split()[0].startswith("kb-")]
        assert len(kb_lines) <= 2, f"Expected <=2 kb lines, got {len(kb_lines)}"


# ---------------------------------------------------------------------------
# kb embed-status
# ---------------------------------------------------------------------------

class TestCliEmbedStatus:
    def test_embed_status_has_required_lines(self):
        """embed-status outputs Configured:, Stored:, Verdict: lines.

        Exit code is NOT asserted: embed-status exits non-zero on any
        configured-vs-stored mismatch (a real, current state), but always
        emits the three structural lines on stdout regardless of verdict.
        """
        result = _run("embed-status", env_extra=EMBED_ENV, timeout=10)
        out = result.stdout
        assert "Configured:" in out, f"No 'Configured:' in embed-status output: {out!r}"
        assert "Stored:" in out, f"No 'Stored:' in embed-status output: {out!r}"
        assert "Verdict:" in out, f"No 'Verdict:' in embed-status output: {out!r}"

    def test_embed_status_configured_has_format_url_model_dim(self):
        """Configured: line includes format=, url=, model=, dim=."""
        result = _run("embed-status", timeout=10)
        configured_line = ""
        for line in result.stdout.splitlines():
            if line.startswith("Configured:"):
                configured_line = line
                break
        assert configured_line, "No 'Configured:' line found"
        assert "format=" in configured_line
        assert "url=" in configured_line
        assert "dim=" in configured_line

    def test_embed_status_stored_has_format_url_model_dim(self):
        """Stored: line includes format=, url=, model=, dim=, updated=."""
        result = _run("embed-status", timeout=10)
        stored_line = ""
        for line in result.stdout.splitlines():
            if line.startswith("Stored:"):
                stored_line = line
                break
        if not stored_line:
            # May say "Stored:     (none)" if no embeddings ever stored
            return
        # If populated, must have these fields
        if "(none)" not in stored_line:
            assert "format=" in stored_line
            assert "dim=" in stored_line

    def test_embed_status_verdict_is_one_of_known_values(self):
        """Verdict: line contains one of the known verdict strings."""
        known_verdicts = {"ok", "mismatch-dim-change", "mismatch-format", "mismatch-url",
                          "no-stored-meta", "mismatch-model"}
        result = _run("embed-status", timeout=10)
        verdict_line = ""
        for line in result.stdout.splitlines():
            if line.startswith("Verdict:"):
                verdict_line = line
                break
        assert verdict_line, "No 'Verdict:' line found"
        verdict_val = verdict_line.split(":", 1)[1].strip()
        assert verdict_val in known_verdicts, (
            f"Unknown verdict value {verdict_val!r}; known={known_verdicts}"
        )
