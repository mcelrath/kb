#!/usr/bin/env python3
"""PostToolUse recorder for the read-index (Phase B). Never blocks (exit 0 always).

- Read result      -> mark the file READ for this session.
- Task (sub-agent) -> mark every source file the report MENTIONS (file path, optionally
  file:line) as mentioned-unread, so the Phase-C gates can require the dispatcher to read
  them before modifying/claiming about them.

Bounded + fail-open: a slow/broken recorder must never disrupt a tool call.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
try:
    import read_index as ri
except Exception:
    sys.exit(0)

# Source-ish file path with a known extension, optional :line. Bounded token charset.
_PATH_RX = re.compile(
    r"(?:[\w./~-]+/)?[\w.-]+\.(?:py|rs|ts|tsx|js|jsx|mjs|go|c|h|cpp|cc|hpp|java|rb|lua|sh|lean|toml)"
    r"(?::\d+)?"
)
_MAX_PATHS = 100


def _extract_paths(text: str) -> list[str]:
    out, seen = [], set()
    for m in _PATH_RX.findall(text or ""):
        p = m.split(":", 1)[0]  # strip :line
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= _MAX_PATHS:
            break
    return out


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

    tool = d.get("tool_name")

    if tool == "Read":
        fp = (d.get("tool_input") or {}).get("file_path")
        if fp:
            ri.mark_read([fp])
        return 0

    if tool == "Task":
        resp = d.get("tool_response") or d.get("tool_result") or ""
        text = resp if isinstance(resp, str) else json.dumps(resp)
        paths = _extract_paths(text)
        if paths:
            ri.mark_mentioned_files(paths)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
