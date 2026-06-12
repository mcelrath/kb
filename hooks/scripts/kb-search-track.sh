#!/bin/bash
# PostToolUse hook tracking kb search activity and seen KB IDs.
#
# Two jobs:
#  1. Set the -searched flag (gate for kb-search-gate.sh before Edit/Write)
#  2. Append any kb-IDs in the command output to ${SESSION_ID}-kb-seen so
#     that subsequent kb search calls automatically exclude them (via
#     _load_session_seen_ids() in kb.py).  This covers:
#       - kb add output: "Added: kb-XXXXXX"  (prevents self-echo on next search)
#       - kb search/list output: result IDs  (prevents re-showing seen results)
#       - kb get output: the fetched ID      (dedupe-kb-get.sh also writes this)

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "${PLUGIN_ROOT}/hooks/scripts/lib/state.sh"

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

# Read session ID from PPID mapping (set by history-isolation.sh)
SESSION_FILE="$STATE_DIR/session-$PPID"
if [[ ! -f "$SESSION_FILE" ]]; then
    echo "WARNING: Session file $SESSION_FILE not found (PPID=$PPID)"
    exit 0
fi
SESSION_ID=$(cat "$SESSION_FILE")

# CLI kb command via Bash
if [[ "$TOOL_NAME" == "Bash" ]]; then
    CMD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

    # Only act on kb commands
    if ! echo "$CMD" | grep -qE '(^|[[:space:];&|`(])(~/\.local/bin/)?kb[[:space:]]+(search|list|get|add|related)\b'; then
        exit 0
    fi

    # Mark that a search was done (for kb-search-gate.sh)
    if echo "$CMD" | grep -qE '(^|[[:space:];&|`(])(~/\.local/bin/)?kb[[:space:]]+search\b'; then
        touch "$STATE_DIR/${SESSION_ID}-searched"
    fi

    # Extract kb IDs from stdout and append to seen file
    KB_SEEN_FILE="$STATE_DIR/${SESSION_ID}-kb-seen"
    echo "$INPUT" | python3 -c "
import sys, json, re
try:
    data = json.load(sys.stdin)
    r = data.get('tool_result', {})
    stdout = r.get('stdout', '') or ''
    stderr = r.get('stderr', '') or ''
    text = stdout + stderr
    ids = re.findall(r'\bkb-\d{8}-\d{6}-[0-9a-f]{6}\b', text)
    for i in set(ids):
        print(i)
except Exception:
    pass
" >> "$KB_SEEN_FILE" 2>/dev/null

fi

# Task delegation to kb-research agent (agent will call kb_search in its session)
if [[ "$TOOL_NAME" == "Task" ]]; then
    SUBAGENT_TYPE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})
    print(tool_input.get('subagent_type', ''))
except:
    pass
" 2>/dev/null)

    if [[ "$SUBAGENT_TYPE" == "kb-research" ]]; then
        touch "$STATE_DIR/${SESSION_ID}-searched"
    fi
fi

exit 0
