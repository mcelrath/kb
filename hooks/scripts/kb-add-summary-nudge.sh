#!/bin/bash
# PreToolUse(Bash) advisory nudge: remind caller to pass --summary to kb add.
# NEVER blocking (exit 0 always). Emits to stderr so it appears as advisory context.
#
# The Claude Code hook protocol delivers a JSON payload on stdin with the tool
# name at `tool_name` and the shell command at `tool_input.command` — NOT env
# vars and NOT a top-level `command`. Parse it correctly (the old version read
# $TOOL_NAME, never set, so it early-exited every time, and `d['command']`,
# which is never present).
RAW=$(cat 2>/dev/null)
printf '%s' "$RAW" | python3 -c '
import sys, json, re
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if d.get("tool_name") != "Bash":
    sys.exit(0)
cmd = (d.get("tool_input") or {}).get("command", "") or ""
if re.search(r"(kb add|kb\.py add|\.local/bin/kb add)", cmd) and "--summary" not in cmd:
    sys.stderr.write(
        "[KB NUDGE] kb add without --summary detected. Convention: always pass "
        "--summary \"<one sentence>\" — you wrote the finding, write its summary "
        "in the same turn. It appears in search results and is far better than "
        "the extractive fallback.\n"
    )
'
exit 0
