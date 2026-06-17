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


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0
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
    sys.stderr.write(
        f"READ-INDEX BLOCK: about to modify {ap} which you have NOT read this session.\n"
        "Read it first (Read tool) so your edit is grounded in the actual current content,\n"
        "then retry. (Do not edit from a sub-agent's or memory's description of the file.)\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
