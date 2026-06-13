#!/usr/bin/env python3
"""
Regression proof for kb-891 guard.

Tests:
  (a) non-interactive + NO --config-dir + provider set  -> REFUSES (exit 2, writes nothing)
  (b) non-interactive + --config-dir <TEMPDIR>          -> SUCCEEDS, writes tempdir/settings.json
                                                           real ~/.claude/settings.json UNCHANGED
  (c) --project TAG path                                -> still works (not blocked)

Run as:  python tmp/cfg891/proof.py
(no args; all sub-cases use pipes / temp dirs so real ~/.claude is never touched)
"""
import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

# Make sure we can import kb.configure from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kb.configure import configure_main

# ------------------------------------------------------------------
# Snapshot real ~/.claude/settings.json  (mtime + sha256)
# ------------------------------------------------------------------
real_settings = Path.home() / ".claude" / "settings.json"

def snapshot(p: Path) -> str:
    if not p.exists():
        return "<absent>"
    content = p.read_bytes()
    return hashlib.sha256(content).hexdigest()

real_before = snapshot(real_settings)
real_mtime_before = real_settings.stat().st_mtime if real_settings.exists() else None
print(f"[snapshot] real settings sha256 before: {real_before[:16]}...")

errors: list[str] = []

# ------------------------------------------------------------------
# (a) Non-interactive, NO --config-dir, provider supplied -> REFUSE
# ------------------------------------------------------------------
print("\n--- (a) non-interactive, no --config-dir, provider=ollama-local ---")

# We are running under a pipe (stdout/stdin not a tty in pytest/subprocess),
# but to be certain we patch isatty.  Also set CLAUDECODE to simulate agent.
import unittest.mock as mock

ns_a = argparse.Namespace(
    install_server=False,
    config_dir=None,          # <-- NOT passed
    project=None,
    provider="ollama-local",
    model="qwen3-embedding:0.6b",
    dim=1024,
    format="openai",
    url="http://localhost:11434/v1/embeddings",
    summary_mode="extractive",
    key=None,
    reembed=False,
    db=None,
)

with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
    with mock.patch("sys.stdin") as mock_stdin, mock.patch("sys.stdout") as mock_stdout:
        mock_stdin.isatty.return_value = False
        mock_stdout.isatty.return_value = False
        rc_a = configure_main(ns_a)

print(f"  exit code: {rc_a}  (expected 2)")
if rc_a != 2:
    errors.append(f"(a) expected exit 2, got {rc_a}")

# Confirm nothing was written
real_after_a = snapshot(real_settings)
if real_after_a != real_before:
    errors.append(f"(a) real ~/.claude/settings.json was MODIFIED! before={real_before[:16]} after={real_after_a[:16]}")
else:
    print("  real ~/.claude/settings.json UNCHANGED ✓")

# ------------------------------------------------------------------
# (b) Non-interactive + --config-dir <TEMPDIR> -> SUCCEEDS
# ------------------------------------------------------------------
print("\n--- (b) non-interactive + explicit --config-dir <tempdir> ---")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    ns_b = argparse.Namespace(
        install_server=False,
        config_dir=str(tmp_path),   # <-- EXPLICITLY passed
        project=None,
        provider="ollama-local",
        model="qwen3-embedding:0.6b",
        dim=1024,
        format="openai",
        url="http://localhost:11434/v1/embeddings",
        summary_mode="extractive",
        key=None,
        reembed=False,
        db=None,
    )

    with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}), \
         mock.patch("kb.configure._write_config_toml"), \
         mock.patch("kb.configure._seed_embedding_meta_if_needed", return_value={}):
        with mock.patch("sys.stdin") as mock_stdin, mock.patch("sys.stdout") as mock_stdout:
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False
            rc_b = configure_main(ns_b)

    print(f"  exit code: {rc_b}  (expected 0)")
    if rc_b != 0:
        errors.append(f"(b) expected exit 0, got {rc_b}")
    else:
        written = tmp_path / "settings.json"
        if written.exists():
            print(f"  {written} written ✓  ({written.stat().st_size} bytes)")
        else:
            errors.append(f"(b) {written} was NOT written")

real_after_b = snapshot(real_settings)
if real_after_b != real_before:
    errors.append(f"(b) real ~/.claude/settings.json was MODIFIED by tempdir run!")
else:
    print("  real ~/.claude/settings.json UNCHANGED ✓")

# ------------------------------------------------------------------
# (c) --project TAG path -> not blocked (per-project path)
# ------------------------------------------------------------------
print("\n--- (c) --project TAG with --config-dir tempdir ---")

with tempfile.TemporaryDirectory() as tmpdir2:
    tmp2 = Path(tmpdir2)
    # Create minimal project dir structure
    (tmp2 / ".beads").mkdir()

    ns_c = argparse.Namespace(
        install_server=False,
        config_dir=None,           # no global config_dir — not used in project path
        project="test-project",
        project_dir=str(tmp2),
        enable_tracker=False,
        db_path_override=None,
        key=None,
        provider=None,
        model=None,
        dim=None,
        format=None,
        url=None,
        summary_mode=None,
        reembed=False,
        db=None,
    )

    with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
        with mock.patch("sys.stdin") as mock_stdin, mock.patch("sys.stdout") as mock_stdout:
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False
            rc_c = configure_main(ns_c)

    print(f"  exit code: {rc_c}  (expected 0)")
    if rc_c != 0:
        errors.append(f"(c) expected exit 0, got {rc_c}")

real_after_c = snapshot(real_settings)
if real_after_c != real_before:
    errors.append(f"(c) real ~/.claude/settings.json was MODIFIED!")
else:
    print("  real ~/.claude/settings.json UNCHANGED ✓")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n--- Summary ---")
print(f"real ~/.claude/settings.json sha256 before: {real_before[:16]}...")
print(f"real ~/.claude/settings.json sha256 after:  {real_after_c[:16]}...")
if real_before == real_after_c:
    print("REAL CONFIG HASH UNCHANGED ✓")
else:
    print("REAL CONFIG HASH CHANGED ✗")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("\nAll (a)/(b)/(c) passed ✓")
    sys.exit(0)
