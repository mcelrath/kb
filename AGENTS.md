# Agent Instructions

This project uses **`kbt`** (the kb-native issue tracker) for issue tracking.
`kbt` needs no external database or services — it stores issues in the local kb
SQLite db and works offline.

## Quick Reference

```bash
kbt ready                  # Find available work (unblocked issues)
kbt show <id>              # View issue details
kbt update <id> --claim    # Claim work atomically
kbt close <id>             # Complete work
```

## Knowledge base — `kb`

Record durable findings and search prior work before reimplementing. The commands
are identical on every harness.

**Search first, then add:**
```bash
kb search "<topic>"                                  # semantic; run unfiltered first, then narrow with -p <project>
kb add "<finding>" -t <type> -p <project> --summary "<one dense sentence>"
```
Types: `success | failure | experiment | discovery | correction`. You write the
`--summary` yourself (one sentence) — it is what shows in search results, so make it
dense and specific.

## Agent bridge — `kb bridge`

Coordinate directly with the other agents on this host. Your sender id is inferred —
just use `kb`. Messages addressed to you (or broadcast to `all`) are injected into your
context every turn. Treat a message as a TASK for you when it is addressed to **you**,
or to `all` and names you; otherwise it is background context.

```bash
# Join once, at the start of your session:
kb bridge announce <your-id> "<what you're working on>" "<what you can help peers with>"

# Send a message (add --needs-reply when you want an answer):
kb bridge send <recipient> "<subject>" --body "<text>"

# Reply — closes a message that was sent to you with --needs-reply:
kb bridge send <sender> "re: <subject>" --reply <message-id> --body "<text>"

# Read your mail on demand (it is usually auto-injected each turn):
kb bridge recv
```
Reply to every message marked `--needs-reply`, using `--reply <its-id>`.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Issue Tracking with `kbt`

**IMPORTANT**: This project uses **`kbt`** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why kbt?

- Dependency-aware: track blockers and `discovered-from` relationships between issues
- Self-contained: issues live in the local kb SQLite db — no external service, works offline
- Agent-optimized: `--json` output, ready-work detection, semantic `kbt search`
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
kbt ready --json
```

**Create new issues** (note: `--title` is required; `kbt create` prints `Created: <id>`):

```bash
kbt create --title "Issue title" --description "Detailed context" --type task --priority 2
kbt create --title "Issue title" --description "Context" --priority 1 --deps discovered-from:<parent-id>
```

**Claim and update:**

```bash
kbt update <id> --claim
kbt update <id> --status in_progress
kbt update <id> --notes "progress note"
```

**Complete work:**

```bash
kbt close <id> --reason "Completed"
```

### Migrating an existing bd/dolt project to kb

If this host still runs bd/dolt, move a project's issues into the kb-native
tracker in one shot:

```bash
kbt bead-migrate            # export dolt → import kb → verify → write .kbt marker → archive+remove .beads/
kbt bead-migrate --dry-run  # preview only; mutates nothing
```

It aborts (no marker, `.beads/` untouched) if the export is truncated or its
fidelity does not match the live dolt issue count. After migrating, `kbt`
resolves to the kb backend via the per-project `.kbt/config.toml` marker.

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `kbt ready` shows unblocked issues
2. **Claim your task atomically**: `kbt update <id> --claim`
3. **Work on it**: implement, test, document
4. **Discover new work?** Create a linked issue:
   - `kbt create --title "Found bug" --description "Details" --priority 1 --deps discovered-from:<parent-id>`
5. **Complete**: `kbt close <id> --reason "Done"`

### Read commands

```bash
kbt list --status open        # all open issues
kbt blocked                   # blocked issues + their blockers
kbt search "<query>"          # semantic search (FTS fallback when offline)
kbt dep add <issue> <depends-on>
kbt show <id> --json          # --json works on read commands (show/list/ready/blocked/children/dep list)
```

### Important Rules

- ✅ Use `kbt` for ALL task tracking
- ✅ Use `--json` on read commands for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `kbt ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

> **Migrating from bd/beads?** `python -m kb.bd_import <bd-export.json>` imports an
> existing `bd export --json` into the kbt issues tables. That importer is the
> only bd touchpoint kb ships.

For more details, see README.md.

## Planning & the expert-review gate

Plans are authored in **native plan mode** (Shift+Tab) — the plan-mode harness owns the
plan file at `~/.claude/plans/<slug>.md` and writes its own active plan file with no
permission prompt (that is the authoring surface). The kb plugin then GATES approval on
review: a `PreToolUse(ExitPlanMode)` hook (`plan-review-hook.py`) blocks ExitPlanMode until
the plan carries a recorded **APPROVED** (or **APPROVED-WITH-REVISIONS**) verdict, keyed to
a sha256 of the exact normalized plan text (any edit re-blocks).

Flow:

```bash
# 1. Author the plan in plan mode (Shift+Tab).
# 2. Review it BEFORE exiting plan mode — the gate denies ExitPlanMode otherwise:
Task(subagent_type="kb:expert-review", prompt="FULL REVIEW: epic=<id> plan=<plan_path> project_root=<root>")
#    REJECTED -> revise (the hash changes; the gate stays closed). APPROVED* -> the agent records the marker.
# 3. ExitPlanMode — now allowed; on approval the plan auto-mirrors (PostToolUse hook) to
#    <project-root>/.kb/plans/PLAN-<slug>.md, committed alongside the code, and nudges step 4.
# 4. /decompose-tasks <plan> (parent-run) -> kbt epic + child tasks -> you verify -> /dispatch <epic>.
```

The verdict marker is a transport-agnostic core any agent host can call:

```bash
kb plan-review status <plan>   # stored verdict JSON for this plan's hash, or 'none'
kb plan-review hash <plan>     # sha256 of the normalized plan text
kb plan-review record <plan> --verdict APPROVED|REJECTED --synthesis "..." \
    --project-root <root> --epic-id <id> [--blocking "..." ...]
```

The verdict lives in the marker + kbt — NOT in the plan filename. Do not use `.approved`/suffix
filename markers. Every deferred / follow-up item in a plan MUST be a real kbt issue (created
before review, `--deps discovered-from:<epic-id>`); the expert-review deferral audit REJECTS plans
that defer work in free text without a tracker-ID.

## Landing the Plane (Session Completion)

**When ending a work session**, complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - tests, linters, builds
3. **Update issue status** - close finished work, update in-progress items
4. **PUSH TO REMOTE** - this is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - clear stashes, prune remote branches
6. **Verify** - all changes committed AND pushed

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- If push fails, resolve and retry until it succeeds
