---
name: persona
description: List the available session personas, or select/activate one for this agent. Invoke as /persona (list) or /persona <name>[-<suffix>] (adopt). The bridge ID IS the persona name; adopting one pins it, claims the bridge id (announce --steal), and loads the persona's binding operating role (re-injected each SessionStart by the kb plugin's session-persona hook so it survives compaction). The kb plugin owns the persona+bridge subsystem.
---

# persona — list or adopt your session persona

Personas live at `<project>/.claude/agents/personas/<name>.md` (a project supplies its own
set, via `kb:project-setup`), with `~/.claude/agents/personas/` as a cross-project fallback.
The **bridge ID IS the persona name** — no mapping layer (no `architect`→`archie` alias
table; the file name IS the canonical id). Append a suffix to run multiple INSTANCES of the
SAME persona (`tip-mathlib`, `terry-nemotron`); the base (before the first `-`) selects the
file, the full id is the distinct bridge id. A suffix is ONLY for same-persona instances — a
genuinely DIFFERENT role gets its OWN base name and its own file (do not encode a variant role
as a suffix; that file would never load, because the base resolves to the existing persona).
The kb plugin's `session-persona.sh` SessionStart hook re-injects the active persona on every
start/compact.

## What to do

Run the helper — it does list-or-adopt in ONE call (dir-resolution, validation, session pin,
bridge `announce --steal`, and emitting the COMPOSED role). Do NOT improvise `ls`/`find`/`cat`;
the script handles all of it. One Bash call:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/skills/persona/persona.sh" "<name-or-empty>"
```

- **No argument** → it prints the available personas. Relay them; tell the user `/persona <name>`.
- **Argument** (`<name>` or `<base>-<suffix>`, lower-case; base = before first `-`) → it pins the
  session, claims the bridge id with `--steal` (explicit adoption = takeover; the SessionStart
  re-announce never steals, kb-72f717.3), and prints the **composed operating role** under
  `----- composed operating role -----`: archetype L1 (`skills/persona/archetypes/<archetype>.md`)
  + augmentation L2 (the `augmentation:` file, or the persona body) + any instance body. **READ
  that block IN FULL and adopt it as your binding role**; read any files it references in full.
  Then confirm to the user: active persona, bridge id (with suffix), re-loads after compaction.
  On `Unknown persona`, relay the valid names it lists.

The `--role` for the bridge announce comes from the persona file's own `role:`/`archetype:`
frontmatter (the script reads it). After adoption, `kb bridge recv`/`send` and the next idle
watcher resolve the full id; the persona's mail + records trail are now this session's.
