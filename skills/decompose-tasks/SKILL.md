---
name: decompose-tasks
description: Turn an APPROVED plan into a kbt epic + child tasks, ready to dispatch. Invoke as /decompose-tasks <plan-file> AFTER a plan has passed kb:expert-review and ExitPlanMode approval. The PARENT agent runs this (not a subagent) so it keeps the plan context to orchestrate the subsequent /dispatch. It creates the epic + tasks, STOPS, has you VERIFY the decomposition, then nudges /dispatch — it never auto-dispatches. Bridges native plan mode to the kbt/dispatch execution model.
---

# decompose-tasks — approved plan → kbt epic + tasks → (you check) → /dispatch

Usage: `/decompose-tasks <plan-file>` (the approved plan, e.g. the ExitPlanMode plan file or
`<project>/.kb/plans/PLAN-<slug>.md`).

Run this as the **parent agent**, AFTER ExitPlanMode approval (plan mode is over, so writes are
allowed). The kbt/`/dispatch` model needs an epic with child tasks; native plan mode produces
neither. This skill fills that gap. It does NOT review the plan (kb:expert-review already did) and
does NOT dispatch — it creates the work breakdown and hands you a checkpoint.

## Protocol

1. **Read the approved plan IN FULL.** Identify its phases/sections, the files each touches, and
   the dependencies/ordering between them.
2. **Create the epic** (project tag from the repo's `.claude/kb-project.json` or CLAUDE.md):
   ```
   kbt create --type epic --prefix <tag> --title "Plan: <slug>" --design-file <plan-file>
   ```
3. **Create one child task per plan phase/unit of work**, linked to the epic:
   ```
   kbt create --type task --prefix <tag> --deps parent-child:<epic-id> --title "<phase>: <what>"
   ```
   - One task per discrete, independently-verifiable unit (a phase, a file-cluster change).
   - **Sequence (do NOT parallelize) tasks that edit the SAME files** — add an ordering note or a
     dependency so /dispatch runs them in waves, not concurrently.
   - Carry over any follow-ups the plan listed (`--deps discovered-from:<epic-id>`).
4. **STOP. Do NOT dispatch.** Print the epic id + the full task list (id, title, deps).
5. **VERIFY the decomposition (MANDATORY — this is the safety gate, not optional prose).** Confirm
   ALL of:
   - [ ] **Coverage**: every phase / unit of work in the plan maps to at least one task; no plan
         section is left without a task (no orphan phases).
   - [ ] **No invention**: no task adds scope the plan does not describe.
   - [ ] **File-conflict sequencing**: every pair of tasks that edit the same file is ordered
         (dependency or explicit note), never left to run in parallel.
   - [ ] **Granularity**: each task is dispatch-sized — independently implementable and verifiable
         by one agent in one pass; split anything that bundles unrelated changes.
   - [ ] **Dependencies**: prerequisite tasks (schema/migration before consumers, etc.) precede
         their dependents.
   If any check fails, FIX the tasks (`kbt update` / create / `kbt dep`) and re-verify before
   proceeding. State explicitly that all five checks passed.
6. **Nudge dispatch** (you decide when): print
   `Decomposition verified. Run: /dispatch <epic-id>`
   Do not auto-run it — the human eyeballs the breakdown first.

## Rules

- **You (the parent) verify — there is no separate reviewer for the decomposition.** The plan was
  already expert-reviewed; the task breakdown is your responsibility, checked here.
- Reuse `/dispatch` (`skills/dispatch`) for execution — this skill only produces the epic+tasks.
- Never auto-dispatch; never re-review the plan; never edit the plan content (its hash gates the
  approval — changing it re-blocks ExitPlanMode).
- If the plan has no discrete phases (a trivial one-task change), create a single task and say so.
