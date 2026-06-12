#!/bin/bash
# PreToolUse hook for Bash - deduplicates `kb get <id>` calls within a session.
# On the second call for the same ID, injects a warning and the cached summary
# rather than blocking (the full content is still fetched, but the agent is told
# it already has this entry).

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "${PLUGIN_ROOT}/hooks/scripts/lib/state.sh"

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)
[[ "$TOOL_NAME" != "Bash" ]] && exit 0

CMD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# Match: kb get <id> (with or without ~/.local/bin/ prefix, with or without --raw)
KB_ID=$(echo "$CMD" | python3 -c "
import sys, re
cmd = sys.stdin.read().strip()
m = re.search(r'(?:^|[;&|]\s*)(?:~\/\.local\/bin\/)?kb\s+get\s+(kb-[0-9a-f-]+)', cmd)
print(m.group(1) if m else '')
" 2>/dev/null)

[[ -z "$KB_ID" ]] && exit 0

SESSION_FILE="$STATE_DIR/session-$PPID"
[[ ! -f "$SESSION_FILE" ]] && exit 0
SESSION_ID=$(cat "$SESSION_FILE")

KB_SEEN_FILE="$STATE_DIR/${SESSION_ID}-kb-seen"
touch "$KB_SEEN_FILE"

if grep -qxF "$KB_ID" "$KB_SEEN_FILE" 2>/dev/null; then
    # Already fetched this session — warn but don't block
    echo "NOTE: $KB_ID was already retrieved this session. The content is already in your context — avoid fetching it again." >&2
    # Exit 0: let the call proceed (don't block, just inform)
fi

echo "$KB_ID" >> "$KB_SEEN_FILE"
exit 0
