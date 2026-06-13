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
HERE="$(dirname "$(readlink -f "$0")")"
WATCHER="$HERE/kb-bridge-watch.sh"

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
    echo "KB_BRIDGE_WATCHER_DOWN (stop allowed after 3 blocks): still no live kb-bridge watcher for '$ID'. You are OFF the bridge. Relaunch ASAP with run_in_background:true and NO timeout parameter (omit it — unbounded hold):  bash $WATCHER $ID" >&2
    rm -f "$CTR" 2>/dev/null
    exit 0
fi

echo "KB_BRIDGE_WATCHER_DOWN: no live kb-bridge SSE watcher for '$ID' — you would go SILENT on the bridge and your orchestrator could not reach you.
Relaunch it NOW as its OWN Bash call with run_in_background:true and NO timeout parameter — OMIT timeout entirely. An omitted timeout runs the watcher UNBOUNDED so it holds for DAYS. Do NOT pass timeout:N — that CAPS the task at N (<=10min) and the harness kills it ('failed' exit 144); that cap-kill was the old 144 bug.
  bash $WATCHER $ID
It holds the SSE until a REAL message (prints BRIDGE_WAKE, exits -> relaunch ONE) or the server closes the connection (e.g. kb-server restart; exits empty -> relaunch ONE). Either way keep exactly ONE live, including when done." >&2
exit 2
