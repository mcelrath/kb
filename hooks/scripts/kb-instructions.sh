#!/bin/bash
# UserPromptSubmit hook (kb-d0m): emit the STATIC kb + bridge instructions ONCE per
# session, on a channel that injects identically on BOTH harnesses.
#
# Why UserPromptSubmit and not SessionStart: SessionStart stdout is injected on Claude
# Code but NOT on goose (goose SessionStart = emit(), fire-and-forget; UserPromptSubmit =
# emit_collect(), stdout appended). To reach goose agents the instruction text must ride
# UserPromptSubmit. To avoid per-turn repetition on Claude (where UserPromptSubmit fires
# every turn) we self-gate to ONCE per session_id via a seen-marker. (Probe kb-d0m /
# kb-20260616-014321-384b68 confirmed 5/5 cross-harness safety.)
#
# This carries the STATIC conventions only. Dynamic per-turn content (live peer messages,
# per-prompt semantic findings) stays in bridge-inject.sh / kb-prompt-surface.py.
set -u

INPUT=$(cat 2>/dev/null)
SESSION_ID=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
[ -z "$SESSION_ID" ] && exit 0

# Once-per-session gate. Portable across harnesses (no CLAUDE_* dependency).
STATE_DIR="${KB_STATE_DIR:-${CLAUDE_STATE_DIR:-$HOME/.cache/kb/state}}"
mkdir -p "$STATE_DIR" 2>/dev/null
MARK="$STATE_DIR/${SESSION_ID//[^A-Za-z0-9_-]/_}-instructions-injected"
[ -f "$MARK" ] && exit 0
: > "$MARK" 2>/dev/null

cat <<'INSTRUCTIONS'
=== kb conventions ===
SEARCH FIRST then ADD. kb search "topic" (unfiltered first; then narrow with -p PROJECT).
ALWAYS pass --summary "<one sentence>" to kb add — you wrote the finding, write its summary.
  kb add "content" -t TYPE -p PROJECT --tags T1,T2 --summary "dense one-liner"
Types: success|failure|experiment|discovery|correction
Tags (confidence): proven|heuristic|open-problem  (importance): core-result|technique|detail

=== agent bridge ===
Coordinate directly with peers on this host. Your sender id is inferred — just use `kb`.
Messages addressed to you (or to `all` and naming you) are auto-injected each turn and are
TASKS for you; other traffic is background. Idle reachability is automatic — you are woken
on a directed message.
  kb bridge announce <your-id> "<what you're working on>" "<what you can help with>"
  kb bridge send <to> "<subject>" --body "<text>" [--needs-reply]
  kb bridge send <sender> "re: <subj>" --reply <message-id> --body "<text>"   # answer a --needs-reply
  kb bridge recv                                                              # drain on demand
Reply to every message marked --needs-reply, using --reply <its-id>.
INSTRUCTIONS
exit 0
