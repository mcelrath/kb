#!/bin/bash
# kb bridge watcher — asyncRewake Stop-hook body (kb-ee7).
#
# Registered as a native `asyncRewake` Stop hook: the harness launches this when the
# agent goes IDLE; it holds the kb-server SSE (GET /bridge/watch?id=<id>),
# RECONNECTING through benign closes, and on a real DIRECTED peer message it writes
# the message JSON to STDERR and `exit 2` — which wakes the idle session (asyncRewake
# injects stderr as a system-reminder). Announce frames and heartbeats never wake.
# Agents no longer launch this manually; `kb bridge watch <id>` stays as an escape hatch.
#
# asyncRewake wake contract: ONLY the SCRIPT's exit code 2 wakes. So the read loop
# runs in the MAIN shell via process substitution (`done < <(curl …)`), NOT a
# `curl | while` pipe — a pipe's RHS subshell `exit 2` would be masked by the script's
# real exit and the wake would never fire (kb-ee7 review B2). On a directed frame we
# `exit 2` at top level; benign closes loop+reconnect; nothing else exits.
#
# Single-instance via flock (kb-ee7 review B1): a second invocation (rapid Stops)
# fails to acquire the per-id lock and exits 0. `pgrep` would be a TOCTOU race.
#
# Usage:  kb-bridge-watch.sh <agent-id>
# Env:    KB_SERVER_URL (default http://<hostname>:8765), CLAUDE_STATE_DIR
set -u
ID="${1:?usage: kb-bridge-watch.sh <agent-id>}"

# --- atomic single-instance: hold a per-id lock for the process lifetime ---
STATE_DIR="${CLAUDE_STATE_DIR:-$HOME/.claude/state}"
mkdir -p "$STATE_DIR" 2>/dev/null
LOCK="$STATE_DIR/kb-watch-${ID//[^A-Za-z0-9_-]/_}.lock"
exec 9>"$LOCK"
flock -n 9 || exit 0   # another watcher for this id is alive; fd 9 stays open => lock held

# --- host-relative server URL ---
# The asyncRewake hook runs harness-side (127.0.0.1 reachable); the manual escape-hatch
# runs sandboxed (namespaced loopback -> must use the host alias, which needs the
# kb-server bound 0.0.0.0). $(hostname) works in both when the server binds 0.0.0.0.
HOSTALIAS="$(hostname)"
BASE="${KB_SERVER_URL:-http://${HOSTALIAS}:8765}"
BASE="${BASE//\/\/localhost:/\/\/${HOSTALIAS}:}"
BASE="${BASE//\/\/127.0.0.1:/\/\/${HOSTALIAS}:}"

LAST=""   # last-seen message id; replayed via Last-Event-ID so a reconnect misses nothing
while true; do
    HDR=()
    [ -n "$LAST" ] && HDR=(--header "Last-Event-ID: $LAST")
    # MAIN-shell read loop (process substitution) so the top-level `exit 2` below is
    # the SCRIPT's exit code. Frames are `id: <n>\ndata: <json>\n\n`; `: ping` ignored.
    while IFS= read -r line; do
        case "$line" in
            id:*) LAST="${line#id: }" ;;
            data:*)
                payload="${line#data: }"
                event=$(printf '%s' "$payload" | python3 -c "import sys,json; print((json.load(sys.stdin).get('event') or '').strip())" 2>/dev/null)
                [ "$event" = "announce" ] && continue
                # Real directed message -> WAKE: JSON to stderr (asyncRewake injects it), exit 2.
                printf 'BRIDGE_WAKE %s\n' "$payload" >&2
                exit 2
                ;;
        esac
    done < <(curl -sN --no-buffer --max-time 604800 "${HDR[@]}" "$BASE/bridge/watch?id=$ID" 2>/dev/null 9>&-)
    # ^ 9>&- closes the flock fd in the curl CHILD: process substitution otherwise lets
    # curl inherit fd 9, and an orphaned curl (after the script exits 2 / is timeout-killed)
    # would keep the lock HELD forever -> flock -n fails on every relaunch -> no watcher ->
    # the agent never wakes again. Closing fd 9 in curl releases the lock the moment the
    # script process exits. (kb-ee7 stale-lock-no-process regression.)
    # curl ended with no directed message (benign close / restart / --max-time) -> reconnect.
    sleep 2
done
