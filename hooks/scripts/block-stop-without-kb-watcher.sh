#!/bin/bash
# Stop hook (kb-2os.3) — block a REGISTERED bridge agent from ending its turn
# while its kb-bridge SSE watcher is dead, so it stays reachable on the bridge.
# Subsumes block-stop-without-bridge-watcher.sh: the watcher is now the plugin
# kb-bridge-watch.sh (SSE-exit-on-first-event vs the kb-server), NOT
# `~/.agent-bridge/bridge watch`.
#
# ADVISORY (not blocking): fires ONLY for sessions registered in agents.json.
# It used to exit 2 (hard-block idle until a watcher was live), but run_in_background
# tasks are REAPED at turn boundaries in this harness — the watcher only lives during
# the active turn (when bridge-inject injection already delivers), never into true
# idle — so hard-blocking just forced a futile relaunch loop (relaunch -> instant
# completion at turn-end -> notification -> re-engage -> relaunch). Now it emits a
# one-line reminder and exits 0. Injection is the reliable channel; the watcher is a
# best-effort bonus. See kb-20260613-144048-50539a. (kb-2os; was exit 2)
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

# Watcher not live. ADVISORY only — do NOT block (exit 0). A relaunch is optional;
# it lives only for the current turn (reaped at turn-end), so it adds little over
# injection. If you WANT best-effort idle reachability, launch ONE with
# run_in_background:true AND timeout:600000:  bash $WATCHER $ID
echo "KB_BRIDGE_WATCHER_DOWN (advisory): no live kb-bridge SSE watcher for '$ID'. Injection still delivers peer messages between turns; the idle watcher is best-effort and reaped at turn boundaries. To attempt idle reachability, launch ONE: bash $WATCHER $ID (run_in_background:true, timeout:600000)." >&2
exit 0
