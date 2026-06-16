#!/bin/bash
# asyncRewake Stop-hook launcher (kb-ee7). Registered in hooks.json as a Stop hook
# with asyncRewake:true: the harness runs THIS in the background when the agent goes
# idle. It resolves the agent's bridge id, then execs the watcher (kb-bridge-watch.sh),
# which holds the SSE and exits 2 on a directed peer message -> wakes the idle session.
# Agents never launch a watcher manually. The watcher's flock guarantees one instance
# across rapid Stops; if no id resolves, we exit 0 (re-armed at the next Stop).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

INPUT=$(cat 2>/dev/null)
SID=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
[ -z "$SID" ] && SID="${CLAUDE_SESSION_ID:-}"

# Resolve agent id (mirrors bridge-resume.sh / bridge-inject.sh):
#   persona pin (authoritative) -> agents.json by session_id -> whoami.
ID=""
PIN_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")/.claude/.persona"
if [ -n "$SID" ] && [ -f "$PIN_DIR/session-$SID" ]; then
    ID=$(tr -d '[:space:]' < "$PIN_DIR/session-$SID" 2>/dev/null)
fi
AGENTS="$HOME/.agent-bridge/agents.json"
if [ -z "$ID" ] && [ -n "$SID" ] && [ -f "$AGENTS" ]; then
    ID=$(jq -r --arg sid "$SID" '.agents[] | select(.session_id == $sid) | .id' "$AGENTS" 2>/dev/null | head -n1)
    case "$ID" in "null"|"") ID="" ;; esac
fi
if [ -z "$ID" ] && [ -x "$HOME/.agent-bridge/bridge" ]; then
    ID=$("$HOME/.agent-bridge/bridge" whoami 2>/dev/null | grep '^Effective identity:' | awk '{print $3}')
fi
[ -z "$ID" ] && exit 0   # no id -> no watcher this idle; re-armed at the next Stop

exec bash "$HERE/kb-bridge-watch.sh" "$ID" "$SID"
