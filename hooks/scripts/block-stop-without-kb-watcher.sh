#!/bin/bash
# Stop hook (kb-2os.3) — block a REGISTERED bridge agent from ending its turn
# while its kb-bridge SSE watcher is dead, so it stays reachable on the bridge.
# Subsumes block-stop-without-bridge-watcher.sh: the watcher is now the plugin
# kb-bridge-watch.sh (SSE-exit-on-first-event vs the kb-server), NOT
# `~/.agent-bridge/bridge watch`.
#
# Safety: fires ONLY for sessions whose session_id is registered in agents.json
# (normal sessions are NEVER blocked). 3-block escape so a stuck agent isn't
# hard-locked. Exit 2 = BLOCK the stop (stderr shown to the agent).
set -u
INPUT=$(cat 2>/dev/null)
SESSION_ID=$(printf '%s' "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)

AGENTS_FILE="$HOME/.agent-bridge/agents.json"
[ -f "$AGENTS_FILE" ] || exit 0
ID=""
[ -n "$SESSION_ID" ] && ID=$(jq -r --arg sid "$SESSION_ID" '.agents[] | select(.session_id == $sid) | .id' "$AGENTS_FILE" 2>/dev/null | head -n1)
case "$ID" in ""|"("*|-*|"null") exit 0 ;; esac

CTR="/tmp/claude-bridge-stopblock-${SESSION_ID}-${ID}"

# Watcher alive? The SSE watcher is a `bash kb-bridge-watch.sh <id>` process.
if pgrep -f "kb-bridge-watch.sh ${ID}\b" >/dev/null 2>&1; then
    rm -f "$CTR" 2>/dev/null
    exit 0
fi

# Watcher dead. 3-block escape.
N=0
[ -f "$CTR" ] && N=$(cat "$CTR" 2>/dev/null || echo 0)
case "$N" in (*[!0-9]*) N=0 ;; esac
N=$((N + 1))
echo "$N" > "$CTR"

if [ "$N" -gt 3 ]; then
    rm -f "$CTR" 2>/dev/null
    exit 0
fi

# JSON decision:block — reason is delivered to the AGENT, not dumped to the
# user's terminal. Instruction only, no narrative.
printf '%s' "{\"decision\":\"block\",\"reason\":\"Bridge watcher down. Relaunch as its own background Bash call (run_in_background:true, OMIT the timeout param): kb bridge watch $ID\"}"
exit 0
