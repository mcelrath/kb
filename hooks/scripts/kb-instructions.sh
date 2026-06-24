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

# SINGLE SOURCE of the operational kb/bridge instructions across all harnesses (kb-asf.8).
# This is the condensed runtime mirror of AGENTS.md's "## Knowledge base" + "## Agent bridge"
# sections — keep the two in sync. Per-project AGENTS.md files must NOT copy this block
# (kb-40c): the plugin injects it everywhere, so there is exactly one operational source.
cat <<'INSTRUCTIONS'
=== knowledge base (kb) — durable findings across sessions ===
SEARCH before you ADD; ADD with a summary.
  kb search "<topic>"           # semantic+FTS; run UNFILTERED first, then narrow with -p PROJECT
  kb add "<finding>" -t TYPE -p PROJECT --tags T1,T2 --summary "<one dense sentence>"
  kb get <kb-id>                # full record incl. evidence
  kb correct <old-id> "<new content>" -r "<reason>"   # supersede a wrong/outdated finding
You WROTE the finding, so YOU write --summary (one sentence) — it is what shows in search results.
Types: success | failure | experiment | discovery | correction
Tags:  confidence = proven|heuristic|open-problem ;  importance = core-result|technique|detail
kb-down fallback (embed/LLM server unreachable): write ~/.claude/pending-kb-adds/<UTC>.txt with a
  "# type: / # project: / # tags:" header; `kb flush-pending` drains it. NEVER fall back to a .md file.

=== issue tracking (kbt) — kb-native tracker (local, offline, no external DB) ===
Use kbt for ALL tracking; never markdown TODO lists.
  kbt ready                     # unblocked work
  kbt show <id> [--json]        # full detail + deps
  kbt create --title "..." --description "..." --type task|bug|feature|epic --priority 2
  kbt create --type epic --prefix <tag> --title "Plan: ..." --design-file <plan>   # plan -> epic
  kbt update <id> --claim       # claim atomically ;  --status in_progress ;  --notes "..."
  kbt close <id> --reason "..." ;  kbt list --status open ;  kbt children <id> ;  kbt dep add <a> <b>
Link a child task to its epic with `--deps parent-child:<epic-id>`; link discovered work with
`--deps discovered-from:<parent-id>`. Priority 0 (critical) … 4 (backlog).

=== agent bridge (kb bridge) — coordinate with peers on this host ===
Your sender id is INFERRED — just use `kb`. Messages addressed to you (or to `all` and naming you)
auto-inject each turn and are TASKS for you; other traffic is background. Idle reachability is
automatic — you are woken on a directed message.
  kb bridge announce <your-id> "<what you're working on>" "<what you can help with>"   # join, once
  kb bridge send <to> "<subject>" --body "<text>" [--needs-reply]
  kb bridge send <sender> "re: <subj>" --reply <message-id> --body "<text>"            # answer a --needs-reply
  kb bridge recv                                                                       # drain on demand
Reply to every message marked --needs-reply, using --reply <its-id>.

=== planning — author in plan mode, GATED on kb:expert-review ===
Author plans in native plan mode (Shift+Tab); the harness owns the plan file at ~/.claude/plans/<slug>.md
(its own active plan file writes without a prompt — that is the plan-authoring surface). A
PreToolUse(ExitPlanMode) gate BLOCKS approval until kb:expert-review records an APPROVED (or
APPROVED-WITH-REVISIONS) verdict for the EXACT plan text (sha256-keyed; any edit re-blocks). Full flow:
  1. Plan mode (Shift+Tab) → draft the plan in the harness plan file.
  2. Before ExitPlanMode: Task(subagent_type="kb:expert-review",
       prompt="FULL REVIEW: epic=<id> plan=<plan_path> project_root=<root>")
     REJECTED -> revise + re-review ;  APPROVED* -> the marker is recorded.
  3. ExitPlanMode now passes; on approval the plan auto-mirrors to <project>/.kb/plans/PLAN-<slug>.md
     (committed alongside code) and you are nudged to the next step.
  4. /decompose-tasks <plan> (parent-run) → kbt epic + child tasks → you verify → /dispatch <epic>.
Verdict lives in the marker+kbt, NOT the filename. Every deferred/follow-up item must be a real kbt
issue (--deps discovered-from:<epic>) BEFORE review.
INSTRUCTIONS
exit 0
