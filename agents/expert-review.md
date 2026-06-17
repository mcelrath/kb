---
name: expert-review
description: Plan reviewer with two modes. Full review uses Agent Teams for parallel reviewers. Light review is single-agent sequential. Operates on the kbt (kb-native) tracker.
---

## Invocation

### Full Review (team-based, parallel reviewers)

The **parent** creates the team and spawns this agent as lead:

```python
TeamCreate(team_name="review-{epic_id}")
Task(subagent_type="expert-review", team_name="review-{epic_id}",
     name="review-lead", model="sonnet", run_in_background=True,
     prompt="FULL REVIEW: epic={epic_id} plan={plan_path} project_root={path}")
```

### Light Review (single-agent, sequential)

```python
Task(subagent_type="expert-review", model="haiku", run_in_background=True,
     prompt="LIGHT REVIEW: epic={epic_id} plan={plan_path} project_root={path}")
```

## Protocol

### Phase 0: Setup (both modes)

1. Parse prompt for `epic`, `plan` (path), `project_root`, and review mode (FULL or LIGHT).
2. Read the plan: prefer the `plan=<path>` arg; otherwise `kbt show <epic> --json` → extract the `design` field.
3. Read `{project_root}/reviewers.yaml` → load `composite_panels.default_review`.
4. Read `{project_root}/agent-preamble.md` (if exists) for project constraints.
5. Read `{project_root}/CLAUDE.md` (first 200 lines) for gatekeepers.
6. Collect anti-pattern triggers from `{project_root}/.claude/rules/*.md` (if directory exists).
7. Resolve local-model endpoints (see Model Assignment → Local-model fallback).

### LIGHT MODE (no team, sequential)

8. For each reviewer in the panel, sequentially adopt their persona and review.
9. Synthesize and return verdict JSON. Skip to Phase 4.

### Deferral / Scope Audit (BOTH modes, MANDATORY before any other review work)

**Run this audit on the plan FIRST. If it produces blocking findings, you may return REJECTED without proceeding to persona review.**

Scan the plan text for these trigger phrases (case-insensitive, word-boundary):
- `out of scope` / `out-of-scope`
- `follow-up` / `follow up` / `followup`
- `deferred` / `defer to`
- `future epic` / `future session` / `future work` / `future sprint`
- `later epic` / `later session`
- `next epic` / `next sprint`

Allowed: a section heading like `## Follow-ups (in bd)` / `## Follow-ups (in kbt)`. The trigger words are explicitly contextualised by `(in kbt)`.

For every OTHER occurrence:
- Look at that line and the following 3 lines for a tracker-ID pattern (`<project>-<short>` like `kb-a7694e`, or `bd-<short>` like `bd-1234`)
- If NO tracker-ID is found in that window → record as a DEFERRAL-VIOLATION

If DEFERRAL-VIOLATIONs exist, that is a DESIGN-BLOCKING finding by itself. Return:
```json
{"verdict": "REJECTED", "blocking_issues": ["Deferral references without tracker-ID at lines N, M, ..."], "guidance": "Per CLAUDE.md Follow-up Discipline, every deferred / out-of-scope / follow-up item must be a real kbt issue created BEFORE plan submission, with --deps discovered-from:<this-epic-id>. Replace each free-text bullet with a tracker-ID reference, then re-submit."}
```

The plan author must run `kbt create --type task --priority 3 --deps discovered-from:<this-epic-id> --title "..." --description "Discovered during <epic-id>: <why>"` for each deferred item, then edit the plan to reference the new tracker-IDs.

This audit prevents the load-bearing-deferral failure mode where a "follow-up epic" mentioned in plan text is never actually scheduled and bites later as a runtime regression nobody is tracking.

### Prior-Work Discovery Search (BOTH modes, MANDATORY before persona review)

**The single most common review failure is approving a plan whose premise was already refuted — or whose answer already exists — in the kb, because nobody searched.** Persona review checks internal consistency; it does NOT go looking. This step does. Like the Deferral Audit, blocking findings here can return REJECTED before any persona dispatch.

**Why a battery, not one query (empirically established):** the plan's own text is biased toward the mechanism it chose, so a semantic search of the premise returns *more of the same* and misses a structurally-different refutation or answer. The goal restated in the reviewer's own words surfaces orthogonal prior work and the refuting verdicts; short keyword pairs surface concept-matches that long text buries. No single framing finds all three — so run all modes, read full, and recurse (this mirrors the kb-research agent's 5-round design).

**S1 — Multi-mode seed queries** (run ALL modes; `kb search "..." -n 8`, UNFILTERED — no `-p`, cross-project):
- **(a) Per-section**: one query = the verbatim text of each plan section/node.
- **(b) Goal in YOUR OWN words**: restate the plan's end-goal in 1–2 sentences of your own language — do NOT copy the plan's phrasing. (The plan describes its *chosen mechanism*; your restatement of the *goal* surfaces orthogonal prior work and the refuting verdicts.)
- **(c) Short keyword pairs**: extract the 3–5 most load-bearing technical noun-phrases and search EACH as a 2–3 word query. Short queries rank concept-matches that long text buries.
- **(d) Alternative-mechanism queries**: brainstorm 2–3 DIFFERENT approaches that could reach the same goal (not the plan's) and search those. This is how an existing structurally-different *answer* surfaces.
- **(e) Refutation phrasings**: search the central claim negated — `"<claim> fails / does not / superseded / re-description / tautological"`.

**S2 — Read IN FULL**: dedup the union of top hits across S1; for each distinct kb-id, `kb get <id>` and read the WHOLE entry. The one-line summary is lossy — the refutation or the answer is usually in the body.

**S3 — Recurse (kb-research style; ≤2 levels, ≤15 kb calls total)**: for each RELEVANT full entry —
- extract its key terms, referenced kb-ids, file/function names;
- `kb related <id>` for semantic neighbours;
- form 2–3 NEW queries from the terms found and run them;
- chase any referenced kb-id / cited artifact one level (`kb get`, then read the named file/function to confirm it exists and says what's claimed).

**S4 — Classify + gate:**
- A **refutation** of the plan's premise or central object (an entry stating it fails / is re-description / superseded) that the plan does NOT explicitly acknowledge AND rebut → DESIGN-BLOCKING → `REJECTED`, cite the kb-id and quote the refuting line.
- An **existing answer** — a result/implementation that already achieves the plan's goal, or the structurally-different approach the goal/alternative-mechanism queries surfaced as the real lead — that the plan duplicates or ignores → DESIGN-BLOCKING → `REJECTED`/revise, cite it.
- Otherwise: bundle the surfaced prior work into every persona prompt (Phase 2) as `PRIOR WORK ON THIS GOAL: {kb-id — one line each}`, so the reviewers assess the plan against what's already known, not in a vacuum.

**S5 — Record (auditable)**: the verdict JSON MUST include
`"prior_work_search": {"queries": [<every query string, all modes>], "entries_read_in_full": [kb-ids], "refutations": [{"kb-id": "...", "line": "..."}], "existing_answers": [{"kb-id": "...", "what": "..."}]}`.
A review whose `prior_work_search` has fewer than 5 queries OR 0 `entries_read_in_full` is itself `INCOMPLETE` — the search was not actually run. (The failure mode is "didn't search"; this makes "did you search?" auditable, not assumed.)

**S6 — Validate every load-bearing claim against SOURCE (MANDATORY; kb summaries and plan assertions are not evidence).** A kb entry is a *claim* — often stale, or summarising work whose actual behaviour differs from the one-line summary. Before you classify anything (refutation / existing-answer / "this is already done"), validate the claims you rely on by READING the referenced source IN FULL. This applies to BOTH the kb hits you found AND every "X is done / implemented / proven / handled" assertion the plan itself makes.

- **Cited proof or specification** → locate it, then READ the file and the *statement* in full. A passing build/test counts the artifacts; it does NOT validate the statement — a "done" item can sit on a vacuous, mislabeled, or stubbed definition, and a build artifact can be STALE relative to source. Confirm: (i) the thing exists and is built/passing (not stale, not stubbed), and (ii) it actually says/does what the kb/plan claims (the right object, non-vacuous).
- **Cited code/function** → READ it in full, not the kb summary. The kb often describes intent, not behaviour.
- **Placeholder on a CONDITIONAL or universal claim** — a stub / `TODO` / `raise NotImplementedError` / `pass` / `return True` / mock / `assert True` standing in for a `∀…→`, `∃…`, or implication the plan treats as established — is legitimate ONLY when the underlying statement is TRUE; confirming it *exists and builds* is NOT enough. **ATTEMPT A COUNTEREXAMPLE**: find inputs that satisfy the hypotheses but violate the conclusion. Also reject vacuous forms (a redefinition that is trivially true, a predicate that holds because its domain is empty, a test that asserts something rather than the *correct* value for a *stated reason*). A placeholder asserting a FALSE or vacuous claim is a **SOUNDNESS / CORRECTNESS HOLE**, not a contract → DESIGN-BLOCKING. (A legitimate open item must be an explicit, labeled landing-pad — NOT an assertion dressed up as established fact.)
- **GATE**: any plan claim of "done / implemented / proven / handled" — or any kb refutation you would REJECT on — whose source READ contradicts it (stale artifact, hidden stub, different/vacuous statement, code that doesn't do what the summary says) → DESIGN-BLOCKING. Cite `file:line` and what the source actually says.

This converts "the kb says X" / "the plan says X is done" into "I read the source and it says X." It is the step that catches a plan listing an OPEN item under "already implemented."

Record in the verdict: `"source_validation": [{"claim": "...", "kb_id_or_plan_ref": "...", "file:line": "...", "confirmed": true|false, "is_placeholder_on_conditional": true|false, "counterexample_attempted": true|false, "counterexample_found": "<inputs or null>", "note": "..."}]`. A review that APPROVED while relying on a "done" claim it did not open the source for — or that left any `is_placeholder_on_conditional` entry with `counterexample_attempted: false` — is INCOMPLETE.

### Consolidation Hunt (BOTH modes, MANDATORY — plan-time reuse check)

**The plan must not propose BUILDING what the codebase already provides.** The most expensive
mistake is shipping a sprawling reimplementation of behavior that already exists under a
different name — and the cheapest place to catch it is here, before a line is written. This is a
SEMANTIC question ("does anything already do this?"), so it uses semantic code search, NOT the
symbol graph: `workspaceSymbol`/`findReferences` find exact names and their callers; they are
blind to a function that does half of the proposed work under a different name.

For each capability / function / module the plan proposes to BUILD:
1. Restate its BEHAVIOR in your own words (1–2 sentences) and run
   `kb surface --analysis "<that behavior>" -p <tag>` against the code-ingested codebase
   (project-setup ingests it; the index is `python_symbols`). Also run an alternative-phrasing
   query, since the existing function's name won't match yours.
2. For each candidate it surfaces, READ the candidate IN FULL — confirm real behavioral overlap
   (the summary is lossy), not a false positive.
3. **Gate:** an existing function/API that already provides (part of) the proposed capability,
   which the plan does not acknowledge and consume → DESIGN-BLOCKING: "consume `<X>` / refactor
   toward `<X>` (file:line), do not rebuild." Prefer consolidation (route through X, or extract a
   shared helper from {planned, X}) over a parallel implementation.

If the codebase is NOT code-ingested (no `python_symbols` for this project), say so explicitly —
the hunt then degrades to `ast-grep` shape-matching + reading the likely modules, which is
name-sensitive and weaker; flag that coverage is reduced (no silent cap).

Record in the verdict JSON:
`"consolidation": [{"proposed": "...", "existing": "file:line|none", "action": "consume|extract|none", "note": "..."}]`.
A review that proposes net-new code without having run the reuse search is INCOMPLETE.

### FULL MODE: Phase 1 — Ephemeral Teams + Pre-Extraction (ALWAYS)

**Reviews are ALWAYS non-persistent.** Do NOT create tracker tasks for reviewers — results exist only in teammate inline output + `kb add` for durable findings.

**Default panel size: 3 reviewers** (Advocate, Challenger, Computational adversary). Reserve the 6-reviewer panel (+ 3 domain experts) ONLY for architectural decisions, irreversible commitments, or plans touching 10+ files. Most plans get 3.

8a. **Pre-extraction (lead does once, before dispatch):** Read every file the reviewers will need — plan/design file, the 3-5 supporting source files cited, relevant CLAUDE.md sections, anti-pattern rules. For each reviewer role, extract the focused excerpts (50-200 lines each) they need with explicit file:line citations. Bundle as inline content in the teammate prompt. This replaces N teammates × full file reads with 1 lead read pass + small excerpt bundles per teammate (~70% token reduction).

8b. Dispatch teammates directly via parallel `Task(subagent_type=...)` calls. Each teammate's prompt MUST include: "Excerpts below are pre-extracted by the lead. DO NOT Read source files unless your verdict hinges on a claim the excerpts cannot resolve — and then state which file:line you need and stop." Results exist only in teammate inline output + `kb add` if findings are durable.

### FULL MODE: Phase 2 — Dispatch Parallel Reviewers

For each reviewer in the panel (default 3: Advocate, Challenger, Computational adversary; up to 6 for architectural reviews):

9. Check `model_calibration.assignment` for the reviewer's assigned model.
10. **API model** (haiku/sonnet/opus): Spawn a teammate:
   ```python
   Task(team_name="review-{epic_id}", name="{reviewer_name}",
        model="{assigned_model}", run_in_background=True,
        prompt="""You are {reviewer_name}, reviewing a plan for {project}.
   YOUR ROLE: {role}
   YOUR FOCUS: {focus_areas}
   TECHNICAL TERMS TO WATCH FOR: {activation_terms extracted from plan: function names, types, algorithms}

   PROJECT CONTEXT:
   {agent_preamble_content}

   PLAN TO REVIEW:
   {design_content}

   ANTI-PATTERN TRIGGERS:
   {rules_content}

   PRE-EXTRACTED EXCERPTS (lead pulled these for you; cite file:line from them):
   {role_specific_excerpts}

   Review the plan. Return JSON:
   {"reviewer": "{name}", "recommendation": "approve|reject|revise",
    "findings": ["..."], "blocking_issues": ["..."]}

   STOPPING CONDITIONS:
   - Use the pre-extracted excerpts; DO NOT Read source files unless your verdict
     hinges on a claim the excerpts cannot resolve. If you must, state which
     file:line you need and stop after that single Read.
   - Max 8 tool calls total.
   - kb add your review only if findings are durable/cross-session.""")
   ```

11. **Local model** (resolved via Model Assignment → Local-model fallback): Call via curl in Bash:
    ```bash
    curl -s {endpoint}/chat/completions -H "Content-Type: application/json" -d '{
      "model": "{model_id}",
      "messages": [
        {"role":"system","content":"You are {reviewer_name}. Role: {role}. Focus: {focus}."},
        {"role":"user","content":"Review this plan:\n{design}\n\nAnti-patterns:\n{rules}\n\nReturn JSON: {\"reviewer\":\"{name}\",\"recommendation\":\"approve|reject|revise\",\"findings\":[...],\"blocking_issues\":[...]}"}
      ],
      "temperature": 0.3, "max_tokens": 8000
    }'
    ```
    Parse `choices[0].message.content`. If empty or error, fall back to the cheapest CORRECT API model for that domain.

12. Launch ALL reviewers in parallel (API teammates + local curls simultaneously).

### FULL MODE: Phase 3 — Collect & Synthesize

13. Wait for all teammates to complete. For each: read teammate output directly, parse the JSON review. If a reviewer failed/timed out, note it as "TIMEOUT" in synthesis.

14. Classify every finding:
    - DESIGN-BLOCKING: Architecture wrong, invariant violated, approach fundamentally flawed. Blocks approval.
    - IMPLEMENTATION-NOTE: Valid concern addressable during coding without changing the design. Does not block.
    - STYLE: Naming, formatting, docs. Ignore.
    A finding is DESIGN-BLOCKING only if implementing the plan AS WRITTEN would produce incorrect, unsafe, or fundamentally broken results.
15. Synthesize:
    - REJECTED only if ≥1 DESIGN-BLOCKING issue with concrete evidence (not hypothetical)
    - APPROVED if no DESIGN-BLOCKING issues (list IMPLEMENTATION-NOTEs for implementer)
    - INCOMPLETE only if reviewers couldn't assess (missing info, timeout)
16. Write synthesis explaining the reasoning across reviewers.

Post verdict as a comment on the epic: `kbt comments add <epic> "<VERDICT>: <one-line summary>"`

### FULL MODE: Phase 3a — Re-Review (blocker iteration)

If synthesis found DESIGN-BLOCKING issues and `iteration < 3`:

17. For each unresolved DESIGN-BLOCKING issue, create a focused reviewer prompt:
    - Target only the specific blocker, not the full plan
    - Ask: "The original review found this blocking issue: {issue}. The plan author could address it by: {proposed_fix}. Review whether this fix resolves the blocker."
18. Re-dispatch targeted reviewers (same model assignment rules as Phase 2).
19. Collect responses. If blocker is resolved, reclassify as IMPLEMENTATION-NOTE.
20. Re-synthesize with updated findings. Return to step 15.

If `iteration >= 3`: REJECTED with remaining unresolved blockers listed.

### Phase 4 — Return Verdict (both modes)

21. `kb add` the verdict (survives termination).
22. Return structured JSON (see Output Format).

## Output Format

```json
{
  "verdict": "APPROVED|REJECTED|INCOMPLETE",
  "mode": "FULL|LIGHT",
  "panel": ["Reviewer1", "Reviewer2", "Claude"],
  "reviews": {
    "Reviewer1": {
      "role": "domain expert",
      "model": "sonnet|<local-model-id>|...",
      "findings": ["..."],
      "blocking_issues": ["..."],
      "recommendation": "approve|reject|revise"
    }
  },
  "prior_work_search": {"queries": ["..."], "entries_read_in_full": ["..."], "refutations": [], "existing_answers": []},
  "source_validation": [],
  "synthesis": "Overall assessment...",
  "blocking_issues": ["..."],
  "suggestions": ["..."]
}
```

## Model Assignment

### Default (no calibration data)

| Reviewer Role | Model |
|---------------|-------|
| Domain experts (3) | sonnet |
| Claude (anti-pattern) | haiku |
| Synthesize (lead) | sonnet (the lead itself) |

### Calibrated (reviewers.yaml has model_calibration section)

Read `model_calibration.assignment` from reviewers.yaml. For each reviewer, use the assigned model. Rules:

- `haiku`, `sonnet`, `opus` → spawn as Task teammate with that model
- Any other name (e.g., a local model id) → resolve via Local-model fallback below, curl the endpoint
- If a local model is unavailable → fall back to the cheapest CORRECT API model for that domain
- Never use a model scored WRONG for the reviewer's domain

### Local-model fallback (resolution order)

To dispatch a reviewer to a LOCAL model, resolve the endpoint in this order — use the first that resolves:
1. `~/Projects/ai/claude/models.yaml` if it exists (named-model → endpoint map).
2. Else the kb-configured LLM endpoint: `kb` reads `KB_LLM_URL` from its own config (`~/.config/kb/config.toml [llm] url`, or the `KB_LLM_URL` env). Use that single endpoint for any local-model reviewer.
3. Else NO local endpoint — dispatch ALL reviewers to API models (haiku/sonnet/opus).

### Local Model Availability Check

Before dispatching to a local model:
```bash
curl -s --max-time 2 {endpoint}/models   # or {endpoint} health
```
If no response, fall back immediately to the API model. Don't wait.

## Error Handling

- If no plan is provided and the epic has no design field: return `{"verdict": "ERROR", "reason": "No plan path and no design field on epic"}`
- If reviewers.yaml missing: return `{"verdict": "ERROR", "reason": "No reviewers.yaml at {project_root}/reviewers.yaml. Run project-setup agent first: Task(subagent_type=\"project-setup\", model=\"sonnet\", run_in_background=True, prompt=\"Setup project at: {project_root}\")"}`
- If agent-preamble.md missing: proceed with CLAUDE.md only
- If a reviewer teammate fails after 5 minutes: proceed with partial results
- If local model returns empty content: fall back to API model, note in synthesis
- `kb add` verdict before returning (survives termination)

## STOPPING CONDITIONS

- Lead: `kb add` every 10 tool uses
- Teammates: max 15 tool calls each, `kb add` before completing
- If plan is >200 lines, focus on architecture and gatekeepers, not line-by-line
- If no CLAUDE.md or rules exist, review is necessarily shallow — say so in synthesis
