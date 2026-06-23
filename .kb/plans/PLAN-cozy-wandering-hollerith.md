# Plan: plan-approval → kbt epic → /dispatch bridge (`/decompose-tasks`)

## Context

Native plan mode (ExitPlanMode + the kb-318a8b gate) approves a plan, but the kbt/`/dispatch`
execution model needs an **epic with child tasks** that native plan mode never creates — a model
mismatch. This adds the missing bridge: after an approved plan, a parent-run skill decomposes it
into a kbt epic + tasks, the parent checks that decomposition, then `/dispatch` executes. Epic
creation deliberately lives OUTSIDE expert-review (reviewers review; they don't author tasks, and
a multi-agent review has no single "owner" for it). Decisions locked with the user:
- `/decompose-tasks` is a **skill run by the PARENT** (preserves context for orchestration), run
  **after** ExitPlanMode (so the plan-mode write ban doesn't apply).
- It **stops** after creating the epic+tasks; the **calling agent CHECKS the decomposition**
  (granularity / deps / full coverage of the plan) before nudging `/dispatch`.
- Add an **APPROVED-WITH-REVISIONS** verdict so minor-revision plans dispatch without a full
  re-review. Handoff is a **nudge** (acceptable — agents reliably act on injected directives).

## Flow

1. (plan mode) parent drafts plan → plan-mode file.
2. parent dispatches `Task(kb:expert-review)` → reviews plan TEXT → records
   `APPROVED | APPROVED-WITH-REVISIONS | REJECTED` (content-hashed; subagent write works in plan mode).
3. ExitPlanMode → gate passes on `APPROVED*`, denies on `REJECTED` → user approves → exit plan mode.
4. PostToolUse approval hook injects: "Approved (verdict X). Next: `/decompose-tasks <plan-file>`."
5. (out of plan mode) parent runs `/decompose-tasks` → creates the kbt epic (`--design-file=<plan>`)
   + child tasks → STOPS → **parent verifies the decomposition** against the plan → nudges `/dispatch <epic>`.
6. `/dispatch` (auto mode) → agents implement, dispatcher verifies, tasks close.

## Changes (files)

1. **New skill `skills/decompose-tasks/SKILL.md`** (parent-run). Instructs: read the approved
   plan; `kbt create --type epic --design-file=<plan>`; decompose the plan's phases/sections into
   child tasks (`--deps parent-child:<epic>` + sequencing for same-file conflicts); STOP and print
   the task list; **CHECK** — every plan phase maps to a task, deps are right, no orphans/missing
   coverage, granularity is dispatch-sized; then emit the `/dispatch <epic>` nudge. Reuse the
   dispatch SKILL's conventions; do NOT auto-dispatch.
2. **`kb/cli/commands/plan_review.py`** — add `APPROVED-WITH-REVISIONS` to the `record --verdict`
   choices (the hash/storage path is unchanged).
3. **`hooks/scripts/plan-review-hook.py`** —
   - `_gate`: treat `verdict in ("APPROVED", "APPROVED-WITH-REVISIONS")` as the pass/`ask` path;
     only `REJECTED` denies (current code special-cases only APPROVED).
   - `_mirror` (PostToolUse, on approval): after mirroring, append an `additionalContext` nudge —
     "Approved (verdict X). Run `/decompose-tasks <plan-file>` to create the epic+tasks, check the
     decomposition, then `/dispatch <epic>`."
4. **`agents/expert-review.md`** — document the third verdict: `APPROVED-WITH-REVISIONS` = no
   DESIGN-BLOCKING issues but IMPLEMENTATION-NOTEs the implementer must heed; still passes the gate.
5. **`~/Projects/ai/claude/CLAUDE.md`** (planning section) — document the flow + the auto-mode
   pairing ("after ExitPlanMode: auto-mode for the `/decompose-tasks → /dispatch` path; manual for
   direct edits"), and that epic creation is `/decompose-tasks`, not expert-review.

## Critical files
- `skills/decompose-tasks/SKILL.md` (new), `kb/cli/commands/plan_review.py`,
  `hooks/scripts/plan-review-hook.py`, `agents/expert-review.md`, `~/Projects/ai/claude/CLAUDE.md`.
- Reuse: `skills/dispatch/SKILL.md` (dispatch conventions), `kbt create/dep` (epic+task creation).

## Verification
1. `kb plan-review record - --verdict APPROVED-WITH-REVISIONS …` succeeds; `status` returns it;
   simulate the hook with a captured payload → expect `permissionDecision: ask` (not deny).
2. Live: in plan mode, record `APPROVED-WITH-REVISIONS` → ExitPlanMode passes (asks).
3. `/decompose-tasks` on a sample approved plan → creates an epic + N tasks matching the plan's
   phases (parent-child deps), STOPS, prints the list + the `/dispatch` nudge — no auto-dispatch.
4. End-to-end dogfood: this very plan → expert-review → ExitPlanMode → `/decompose-tasks` →
   check → `/dispatch` builds the feature.
5. Bump `plugin.json`; `claude plugin update kb@kb-local`.

## Follow-ups (in kbt)
- (file before dispatch if any surface) — e.g., whether `/decompose-tasks` should also handle
  re-decompose when a plan is revised (hash changed) without duplicating the epic.
