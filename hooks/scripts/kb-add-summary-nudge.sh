#!/bin/bash
# PreToolUse(Bash) advisory nudge: remind caller to pass --summary to kb add.
# NEVER blocking (exit 0 always). Emits to stderr so it appears as advisory context.

# Only fire on Bash tool calls that contain "kb add" or "kb_add" without --summary
TOOL_INPUT="${TOOL_INPUT:-}"
TOOL_NAME="${TOOL_NAME:-}"

# Only care about Bash tool
[[ "$TOOL_NAME" != "Bash" ]] && exit 0

# Extract command from JSON input (hook receives JSON on stdin or via TOOL_INPUT)
COMMAND=""
if [[ -n "$TOOL_INPUT" ]]; then
    COMMAND="$TOOL_INPUT"
else
    # Try reading from stdin (Claude Code hook protocol: JSON on stdin)
    RAW=$(cat 2>/dev/null)
    COMMAND=$(printf '%s' "$RAW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('command',''))" 2>/dev/null || true)
fi

# Check: contains "kb add" (or kb.py add / kb_add pattern) but NOT --summary
if printf '%s' "$COMMAND" | grep -qE '(kb add|kb\.py add|\.local/bin/kb add)'; then
    if ! printf '%s' "$COMMAND" | grep -q -- '--summary'; then
        echo "[KB NUDGE] kb add without --summary detected. Convention: always pass --summary \"<one sentence>\" — you wrote the finding, write its summary in the same turn. It appears in search results and is far better than the extractive fallback." >&2
    fi
fi

exit 0
