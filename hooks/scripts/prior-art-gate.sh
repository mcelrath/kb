#!/bin/bash

# --- EMBEDDING-DOWN gate (ash:8081): surface hard STOP instead of blind retrieval ---
. "$(dirname "$0")/lib/ash_health.sh" 2>/dev/null || true
if command -v ash_down >/dev/null 2>&1 && ash_down; then
  echo "$ASH_STOP_LINE" >&2
fi

# PreToolUse hook: Task.
# Blocks Task dispatch (non-kb-research) unless kb-research was run this session.
# (The physics Edit/Write prior-art gate was split out to
# secular-constraints/.claude/hooks/physics-edit-prior-art.sh — kb-bp4 P5.)

source "$HOME/.claude/hooks/lib/state.sh"

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

# --- Task dispatch gate (unchanged) ---
if [[ "$TOOL_NAME" == "Task" ]]; then
    SUBAGENT_TYPE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('subagent_type', ''))
except:
    pass
" 2>/dev/null)

    [[ "$SUBAGENT_TYPE" == "kb-research" ]] && exit 0

    SESSION_FILE="$STATE_DIR/session-$PPID"
    [[ ! -f "$SESSION_FILE" ]] && exit 0
    SESSION_ID=$(cat "$SESSION_FILE")

    SEARCHED_FILE="$STATE_DIR/${SESSION_ID}-searched"
    if [[ ! -f "$SEARCHED_FILE" ]]; then
        echo "BLOCKED: Dispatching agent without prior kb-research this session." >&2
        echo "" >&2
        echo "Run kb-research first this session; pass results in dispatch prompt." >&2
        echo "Example: Task(subagent_type='kb-research', model='haiku', prompt='TOPIC: <your topic>')" >&2
        exit 2
    fi
    exit 0
fi

exit 0
