---
name: dispatch
description: Execute a kbt epic by dispatching agents in waves, verifying their work, and looping until done. Invoke as /dispatch <epic-id> when the user wants an approved epic implemented autonomously (the coordinator claims ready child tasks, spawns implementation agents, verifies every diff itself, closes tasks, and commits). Operates on the kbt (kb-native) tracker.
---

# dispatch — execute a kbt epic in waves

Usage: `/dispatch <epic-id>` (a kbt epic id, e.g. `kb-72f717`).

Operates on the **kbt** (kb-native) tracker — NOT bd/dolt. Every tracker command is `kbt`.

## Protocol

You are the dispatch coordinator. Keep agents spinning on the epic until every task is
complete. You do NOT delegate judgment — you verify everything yourself.

### Phase 1: Survey
1. `kbt show <epic-id>` — read the epic, its `--design-file`, and all child tasks.
2. `kbt children <epic-id> --json` — enumerate child tasks; `kbt ready` — which are unblocked.
3. Read every file the epic's design references, IN FULL. You need full context to judge agent work.
4. SEQUENCE BY FILE-CONFLICT: tasks that edit the SAME files must NOT run in parallel — order them; only independent tasks share a wave.

### Phase 2: Dispatch loop
```
while unfinished tasks remain:
  1. IDENTIFY ready, non-conflicting tasks (kbt ready, minus same-file overlaps).
  2. DISPATCH up to 3 agents in parallel (background):
     - One Agent() call per task, run_in_background=true, model: sonnet (Haiku only for pure lookups; never Opus).
     - Prompt MUST include:
       * "Read ~/.claude/agents/preamble.md FIRST" AND, if the project has one,
         "Read <project>/.claude/agents/preamble.md too"
       * the task description from `kbt show <task-id>` + the design-file path
       * "Work in the MAIN tree (NOT a worktree — changes are LOST)"
       * "STOPPING CONDITIONS: <what done looks like>"
       * "run `kb add` with a one-sentence --summary before returning"
       * "`kbt update <task-id> --notes ...` before returning"
     - `kbt update <task-id> --claim` BEFORE dispatching.
     - NEVER isolation:"worktree" (auto-deleted; changes lost).
  3. WAIT for completion notifications.
  4. When an agent returns:
     a. READ EVERY FILE IT TOUCHED in full (`git diff` to see what changed) — summaries describe intent, not reality.
     b. Run tests if they exist (pytest / python3 -m py_compile / project suite).
     c. WRONG -> fix yourself (small) or re-dispatch (large).
     d. GOOD -> `kbt close <task-id>`.
     e. `git add <changed files> && git commit --no-gpg-sign` (bump plugin.json version in the SAME commit for kb-repo changes).
  5. Re-check `kbt ready`: newly-unblocked tasks -> next wave immediately.
  6. Agent >10 min: kill it; do the task yourself or re-dispatch.
```

### Phase 3: Completion
1. `kbt children <epic-id> --json` — verify no open children remain.
2. Run the full test suite if one exists.
3. `kbt close <epic-id>`.
4. Surface discovered-from follow-ups (no orphan deferrals):
   ```bash
   kbt dep list <epic-id> --json | jq -r '.[] | select(.type|test("discovered";"i")) | select(.status!="closed") | "  - \(.id): \(.title)"'
   ```
   If non-empty, report under "Follow-ups (open, discovered during this epic):" and suggest
   "next: `kbt show <id>`, or `/dispatch` the next priority one." Do NOT close the session without showing these.
5. Report: what was done, what was committed, verification findings, and the follow-ups list.

## Critical rules
- **YOU verify, not the agent.** READ the files; never trust a summary.
- **Commit after each verified task**, not in one batch — prevents lost work. Bump plugin.json on kb-repo commits.
- **Keep the loop tight.** Don't pause to ask between waves — it's autonomous execution of an already-approved plan.
- **Fix small problems yourself** rather than dispatching for a 5-line fix.
- **Never >3 agents simultaneously.** Never `isolation:"worktree"`.
- **A task fails twice -> STOP and report**, don't retry indefinitely.
- **Models:** Sonnet for implementation, Haiku for pure lookups only, never Opus for subagents.
