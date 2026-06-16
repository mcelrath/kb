#!/bin/bash
# kb-based bridge watcher (kb-jij.6 / kb-2os.3) — SSE-exit-on-first-event.
#
# Replaces `~/.agent-bridge/bridge watch <id>`. Connects to the kb-server SSE
# endpoint GET /bridge/watch?id=<id>, reads frames LINE-BY-LINE, and on the first
# `data:` frame (a real peer message — `: ping` heartbeats are skipped) prints it
# and EXITS. The exit is the wake: launched via run_in_background, the task-exit
# fires a notification re-engaging an idle agent. Single-shot; fresh subscriber
# starts at the tail (only NEW messages wake), same contract as the old watcher.
#
# Reads via a `while read` loop, NOT `grep -m1`: grep buffers its stdin in blocks
# and never processes a single small SSE frame until the stream closes; `read`
# returns each line as soon as its newline arrives (curl -N writes frames
# immediately), so the wake is prompt.
#
# Usage:  kb-bridge-watch.sh <agent-id>
# Env:    KB_SERVER_URL (default http://ash:8765)
#
# Default host is `ash`, NOT 127.0.0.1: this watcher is launched via the Bash
# tool with run_in_background, so it runs inside the command SANDBOX, whose
# loopback is namespaced (cannot reach the host's 127.0.0.1:8765). The sandbox
# DOES proxy the allowlisted host `ash`, and the kb-server binds 0.0.0.0, so
# http://ash:8765 is reachable from inside the sandbox. (Harness-side hooks run
# OUTSIDE the sandbox and still use the 127.0.0.1 default — kb-2os.3 sandbox fix.)
set -u
# NO signal trap here. An earlier `trap 'exit 0' TERM PIPE HUP INT 16` (meant to
# turn teardown-144s into clean exits) was the bug: it fired on a benign signal
# during normal background operation and exited the script IMMEDIATELY with empty
# output — turning a watcher that should hold ~540s into one that exits in seconds,
# which fed a relaunch loop (and it didn't even reliably prevent the 144). A raw
# bounded curl-SSE with no trap holds correctly (verified: 305s sandboxed, 109s
# not). curl's own --max-time gives the clean quiet-timeout exit; let it be.
ID="${1:?usage: kb-bridge-watch.sh <agent-id>}"
# Resolve the LOCAL host alias. This watcher runs inside the Bash-tool sandbox,
# whose loopback is namespaced — host 127.0.0.1/localhost is UNREACHABLE here even
# though harness-side hooks reach it fine. The LOCAL kb-server (bound 0.0.0.0) IS
# reachable via the host's OWN hostname alias (sandbox-proxied + allowlisted).
# Use $(hostname) so this is HOST-PORTABLE — ash->ash:8765, tardis->tardis:8765.
# Hardcoding `ash` sent every other host's watcher to ASH's server, subscribing to
# the WRONG bridge while local POSTs hit the LOCAL server -> sends never woke it
# (mis-read as a send<->watch wiring gap; it was a cross-host mismatch).
# REQUIREMENT: the local kb-server must bind 0.0.0.0 (not 127.0.0.1-only) so
# <hostname>:8765 is reachable from the sandbox.
HOSTALIAS="$(hostname)"
BASE="${KB_SERVER_URL:-http://${HOSTALIAS}:8765}"
BASE="${BASE//\/\/localhost:/\/\/${HOSTALIAS}:}"
BASE="${BASE//\/\/127.0.0.1:/\/\/${HOSTALIAS}:}"
URL="$BASE/bridge/watch?id=$ID"

# LAUNCH THIS WITH run_in_background:true AND THE timeout PARAMETER OMITTED.
#   - An OMITTED timeout runs the bg task UNBOUNDED, so the watcher holds for DAYS
#     (verified: a no-timeout bg task ran well past the 600000ms param-max with no
#     kill). This is the correct way to run the watcher.
#   - PASSING timeout:N CAPS the task at N (<=600000ms) and the harness KILLS it at
#     the cap (exit 144, "failed") — that cap-kill, not the SSE, was the old "144".
#     So: do NOT pass timeout.
# --max-time 604800 (1 week) is only a connection-freshness backstop; in practice the
# watcher exits earlier — on a real message (prints BRIDGE_WAKE, exit 0) or when the
# server closes the SSE (e.g. kb-server restart; curl exits 0) -> agent relaunches.
# No trap + no harness cap -> only clean exits, never a 144.
curl -sN --no-buffer --max-time 604800 "$URL" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        data:*)
            # SSE data frame = the message JSON.
            payload="${line#data: }"
            # Skip event:announce — announces are non-actionable and are already
            # surfaced between turns by bridge-inject (GET /bridge/messages includes
            # to:all). The watcher wakes ONLY for real directed/actionable messages,
            # so an idle agent isn't re-engaged just to read a peer joining.
            event=$(printf '%s' "$payload" | python3 -c "import sys,json; print((json.load(sys.stdin).get('event') or '').strip())" 2>/dev/null)
            [ "$event" = "announce" ] && continue
            # Real message: surface it + exit (closes the pipe -> SIGPIPE kills
            # curl -> this subshell + the script end). The exit IS the wake.
            echo "BRIDGE_WAKE $payload"
            exit 0
            ;;
        ': ping'|': keepalive')
            # SSE heartbeat comment. Emit a keepalive marker so the background-task
            # harness sees activity and doesn't kill the silent pipeline.
            echo "BRIDGE_PING"
            ;;
    esac
done
exit 0
