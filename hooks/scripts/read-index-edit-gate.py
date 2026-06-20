#!/usr/bin/env python3
"""Edit/Write gate (Phase C, kb-3d347c): block modifying an EXISTING source file this
session has not READ. Block-strength (exit 2). PreToolUse — no loop risk (denies one call;
the agent reads, then retries).

- New file (not on disk) -> allow (nothing to read).
- File read this session -> allow.
- Existing file NOT read  -> BLOCK; the agent must Read it first.

For the Claude harness this is mostly a no-op (it already requires read-before-edit, and our
PostToolUse recorder marks that read); its real value is enforcing the same on harnesses /
sub-agents that do NOT, and on Write-overwrites of unread files. Fail-open.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
try:
    import read_index as ri
except Exception:
    sys.exit(0)

_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "developer__text_editor"}

# The PostToolUse recorder touches the session ledger on EVERY Read. If the
# ledger is missing (recorder never ran) or not touched within this window
# (recorder died mid-session), is_read() is untrustworthy and blocking would be
# an UNBREAKABLE false block — so the gate fails OPEN with a warning instead.
_RECORDER_STALE_SEC = 600


def _recorder_alive() -> bool:
    try:
        ledger = ri.state_path("read-index")
    except Exception:
        return False
    if not ledger or not os.path.isfile(ledger):
        return False
    try:
        import time
        return (time.time() - os.path.getmtime(ledger)) <= _RECORDER_STALE_SEC
    except OSError:
        return False


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0
    # Recover session id from the stdin payload when CLAUDE_SESSION_ID is absent
    # — hook subprocesses spawned after a live `claude plugin update` reload lose
    # the env var, and the PPID fallback can miss; the payload always carries it.
    if not os.environ.get("CLAUDE_SESSION_ID"):
        _sid = d.get("session_id")
        if _sid:
            os.environ["CLAUDE_SESSION_ID"] = str(_sid)

    if d.get("tool_name") not in _EDIT_TOOLS:
        return 0
    ti = d.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("path") or ""
    # goose text_editor: a "create" command makes a new file — nothing to read.
    if ti.get("command") == "create":
        return 0
    if not fp:
        return 0
    ap = os.path.abspath(os.path.expanduser(fp))
    if not os.path.isfile(ap):
        return 0  # new file
    if ri.is_read(ap):
        return 0
    if not _recorder_alive():
        sys.stderr.write(
            "READ-INDEX: read-recorder appears down (session ledger missing or "
            "stale) — cannot verify reads; ALLOWING this edit. Read-before-edit is "
            "unenforced until hooks reload (restart the session).\n"
        )
        return 0
    sys.stderr.write(
        f"READ-INDEX BLOCK: about to modify {ap} which you have NOT read this session.\n"
        "Read it first (Read tool) so your edit is grounded in the actual current content,\n"
        "then retry. (Do not edit from a sub-agent's or memory's description of the file.)\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
