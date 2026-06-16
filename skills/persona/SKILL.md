---
name: persona
description: List the available session personas, or select/activate one for this agent. Invoke as /persona (list) or /persona <name>[-<suffix>] (adopt). The bridge ID IS the persona name; adopting one pins it, claims the bridge id (announce --steal), and loads the persona's binding operating role (re-injected each SessionStart by the kb plugin's session-persona hook so it survives compaction). The kb plugin owns the persona+bridge subsystem.
---

# persona — list or adopt your session persona

Personas live at `<project>/.claude/agents/personas/<name>.md` (a project supplies its own
set, via `kb:project-setup`), with `~/.claude/agents/personas/` as a cross-project fallback.
The **bridge ID IS the persona name** — no mapping layer. Append a suffix to run multiple
instances of one persona (`tip-mathlib`); the base (before the first `-`) selects the file.
The kb plugin's `session-persona.sh` SessionStart hook re-injects the active persona on every
start/compact.

## What to do

1. **Resolve the persona dir** (first that exists): `$(git rev-parse --show-toplevel)/.claude/agents/personas` → `$PWD/.claude/agents/personas` → `~/.claude/agents/personas`.

2. **No argument — LIST.** Show each `*.md` as `name — <first description line>`; tell the user to `/persona <name>`. If the dir is empty/absent, say the project defines no personas.

3. **Argument names a persona** (or `<base>-<suffix>`; lower-case; base = before first `-`):
   1. Confirm `<base>.md` exists; else list valid names.
   2. Pin the id AND claim the bridge id (EXPLICIT adoption = a deliberate action, so it
      `announce --steal`s NOW — the takeover path; the automatic SessionStart re-announce
      never steals, it qualifies-on-conflict, so the steal must happen here, kb-72f717.3):
      ```bash
      FULL_ID="<full-id>"
      D="$(git -C . rev-parse --show-toplevel 2>/dev/null || echo .)/.claude/.persona"
      SID="${CLAUDE_SESSION_ID:-unknown}"
      mkdir -p "$D"; echo "$FULL_ID" > "$D/session-${SID}"
      BR="$HOME/.agent-bridge/bridge"
      [ -x "$BR" ] && "$BR" announce --id "$FULL_ID" --role "<role from <base>.md frontmatter>" \
        --focus "adopted via /persona" --offering "<offering>" --steal </dev/null 2>/dev/null \
        && echo "pinned + claimed bridge id '$FULL_ID' (--steal)"
      ```
      After this, `kb bridge recv`/`send` and the next idle watcher resolve `$FULL_ID`; the
      persona's mail + records trail are now this session's.
   3. **Read `<base>.md` IN FULL** and adopt it as your binding operating role; read any files it references in full.
   4. Confirm: active persona, bridge id (with suffix), and that it re-loads after compaction.
