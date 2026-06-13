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
BASE="${KB_SERVER_URL:-http://ash:8765}"
# This watcher runs inside the Bash-tool sandbox, whose loopback is namespaced —
# host 127.0.0.1/localhost is UNREACHABLE here even though hooks (harness-side)
# reach it fine. The kb-server binds 0.0.0.0, so rewrite a loopback base to the
# `ash` host alias (sandbox-proxied + allowlisted). Non-loopback bases pass through.
BASE="${BASE//\/\/localhost:/\/\/ash:}"
BASE="${BASE//\/\/127.0.0.1:/\/\/ash:}"
URL="$BASE/bridge/watch?id=$ID"

# --max-time 540 (9 min), NOT an infinite hold. Two facts about run_in_background:
#   1. Without an explicit harness timeout: param, a bg task gets a default timeout
#      and is KILLED when it runs past it (signal -> exit 144, "failed") — which
#      silently drops the agent off the bridge. So this MUST be launched with
#      timeout:600000 (the 10-min max). Every 144 we saw was a no-param launch;
#      the timeout:600000 launch held cleanly.
#   2. Even with timeout:600000 there is a ~10-min hard cap, so --max-time 540 keeps
#      the hold UNDER it: curl times out CLEANLY at 540s (exit 28 -> this script
#      exits 0 -> "completed"), and the agent relaunches. Idle agent stays reachable
#      on a ~9-min cycle and never throws a 144. A real message wakes it instantly.
curl -sN --no-buffer --max-time 540 "$URL" 2>/dev/null | while IFS= read -r line; do
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
    esac
done
exit 0
