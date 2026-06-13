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
# Env:    KB_SERVER_URL (default http://127.0.0.1:8765)
set -u
ID="${1:?usage: kb-bridge-watch.sh <agent-id>}"
BASE="${KB_SERVER_URL:-http://127.0.0.1:8765}"
URL="$BASE/bridge/watch?id=$ID"

curl -sN --no-buffer --max-time 86400 "$URL" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        data:*)
            # SSE data frame = the message JSON; surface it + exit (closes the
            # pipe -> SIGPIPE kills curl -> this subshell + the script end).
            echo "BRIDGE_WAKE ${line#data: }"
            exit 0
            ;;
    esac
done
exit 0
