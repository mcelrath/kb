# Knowledge Base (kb)

SQLite + sqlite-vec powered findings database for tracking successes, failures,
experiments, and discoveries across projects — with a Claude Code plugin that
surfaces relevant findings and injects kb conventions automatically, plus `kbt`,
a kb-native issue tracker, and a small HTTP/SSE server (`kb-server`) used by the
plugin and an agent message bridge.

## Features

- **Vector similarity search** via sqlite-vec (semantic retrieval), with **FTS5**
  full-text fallback when the embedding server is unreachable.
- **LLM query expansion** (`--expand`) for improved recall (optional).
- **Supersession chains** for correcting outdated findings.
- **Multi-harness plugin (Claude Code + goose)** — ONE repo, one shared
  `hooks/hooks.json`. The plugin **injects the kb / kbt / bridge instructions into the
  agent's context every session** and surfaces relevant findings per prompt — with **no
  edits to `CLAUDE.md` / `.goosehints` / `AGENTS.md`** and no permission allow-listing.
- **Installs `kb` on PATH for you** — the plugin builds a venv and `pip install -e`s kb
  into it, then drops a `~/.local/bin/kb` wrapper, so the injected `kb …` instructions work
  out of the box (it never clobbers an existing hand-written wrapper).
- **`kbt` issue tracker** — kb-native (local SQLite, offline, no external DB), bd-compatible CLI.
- **Agent bridge (`kb bridge`)** — host-local cross-agent messaging (announce / send /
  reply / recv); peer messages auto-inject into your context, idle sessions are woken on a
  directed message.
- **`kb-server`** — HTTP/SSE endpoints (kb/issue reads + the bridge transport).

There is **no MCP server** — all kb operations go through the `kb` CLI (and, inside
Claude Code, the plugin hooks). References to `kb_mcp.py` or `mcp__knowledge-base__*`
tools are stale.

## Prerequisites

- **Python 3.11+** (3.11, 3.12, 3.13, 3.14 all supported; 3.14+ is slightly faster
  via PEP-649 lazy annotation evaluation but not required).
- **An embedding endpoint.** kb does not run a local embedding model — it calls a
  remote endpoint (llama.cpp `/embedding`, or any OpenAI-compatible `/v1/embeddings`
  such as **Ollama**). For a CPU box: `ollama pull qwen3-embedding:0.6b` (1024-dim).
- Optional: an **LLM completion endpoint** for `--expand` query expansion and
  `local-llm` summaries (kb degrades gracefully without it).
- [`uv`](https://github.com/astral-sh/uv) recommended for the venv.

## Install

```bash
# 1. Clone, build the venv, install deps
git clone <kb-repo-url> kb && cd kb
uv venv --seed                      # uses your default python3 (needs 3.11+)
.venv/bin/pip install -r requirements.txt

# 2. Configure embedding + LLM endpoints (interactive: health-checks both, and
#    offers to install + start the kb-server systemd unit at the end)
.venv/bin/python kb.py configure
#   …or non-interactive, e.g. Ollama on a CPU box:
.venv/bin/python kb.py configure \
  --provider ollama --model qwen3-embedding:0.6b --dim 1024 \
  --format openai --url http://localhost:11434/v1/embeddings \
  --llm-url ""                       # blank/unreachable LLM is fine

# 3. (non-interactive only) install + start the kb-server systemd --user unit
.venv/bin/python kb.py configure --install-server [--server-port 8765]

# 4. Put `kb` AND `kbt` on PATH (~/.local/bin). Agents and the plugin's hooks call
#    them BY NAME. This is one flag:
.venv/bin/python kb.py configure --install-wrappers
#    (it skips any existing hand-written wrapper, and warns if ~/.local/bin isn't on PATH.)
```

> **If you use the Claude Code / goose plugin (below), you can skip steps 1 and 4** — the
> plugin builds its own venv and installs the `kb`/`kbt` wrappers for you on first run. You
> still want step 2 (`kb configure`) to point kb at your embedding endpoint.

`kb configure` writes `~/.config/kb/config.toml` (the source of truth) and mirrors
the non-secret values into the config dir's `settings.json`; any API key goes to
`settings.local.json` (only after `git check-ignore` confirms it's ignored).

## Plugin (Claude Code + goose)

The recommended way to use kb. **One repo is both a Claude Code plugin and a goose
plugin**, sharing a single `hooks/hooks.json`. The hooks build a venv, put
`kb`/`kbt` on PATH, inject the kb/kbt/bridge conventions, and surface findings —
**with no edits to `CLAUDE.md`, `.goosehints`, or `AGENTS.md`**.

```bash
# Claude Code: add the local marketplace (the repo root, containing .claude-plugin/) + install
claude plugin marketplace add /path/to/kb
claude plugin install kb@kb-local
```

```bash
# goose: point goose at the same repo's shared hooks file
#   ~/.config/goose/config.yaml  ->  hooks:  - /path/to/kb/hooks/hooks.json
# Scripts use ${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT}}, so the same hooks run under both harnesses.
```

### Isolating a test/fresh config from your real `~/.claude`

Set **`CLAUDE_CONFIG_DIR`** to a separate directory before launching Claude Code —
your global `~/.claude` (CLAUDE.md, hooks, personas, settings) is then **not**
loaded, and the marketplace + installed plugins live under the isolated dir:

```bash
export CLAUDE_CONFIG_DIR="$HOME/kb-sandbox/.claude"
mkdir -p "$CLAUDE_CONFIG_DIR"
claude plugin marketplace add /path/to/kb
claude plugin install kb@kb-local
# configure into the SAME isolated dir (configure defaults to the real ~/.claude otherwise):
.venv/bin/python kb.py configure --config-dir "$CLAUDE_CONFIG_DIR" --provider ollama …
cd <your-repo> && claude        # CLAUDE_CONFIG_DIR is honored
```

### What the hooks do

On **SessionStart** (in order):

1. **setup-venv.sh** — builds the deps venv at `$CLAUDE_PLUGIN_DATA/venv`, `pip
   install -e`s the package, installs the `kb`/`kbt` wrappers (idempotent, hash-gated;
   needs pypi.org on first run).
2. **env-probe.sh** — confirms/injects `CLAUDE_PLUGIN_ROOT` / `CLAUDE_PLUGIN_DATA`.
3. **session-persona.sh** — restores this session's bridge persona (durable identity).
4. **kb-flush-pending.sh** — drains any queued offline kb-adds.
5. **kb-context.sh** — surfaces recent findings + resume context; emits a `KB-INFRA
   DOWN` warning and falls back to FTS-only if the embedding server is unreachable.
6. **scaffold-check.sh** — flags a missing `reviewers.yaml` and offers project setup.
7. **bridge-resume.sh** — re-attaches the bridge identity (qualifies on session conflict).
8. **session-followups.sh** — surfaces open follow-ups from recently closed epics.
9. **bridge-watch-rewake.sh** — restarts the idle-reachability watcher.

The **kb/kbt/bridge conventions** themselves are injected by **kb-instructions.sh** on
**UserPromptSubmit**, once per session (so a new session always has them, even after
SessionStart context is trimmed). Relevant findings are surfaced per-prompt the same way.

Hooks invoke kb/kbt via the `~/.local/bin` wrappers (or, internally,
`"${CLAUDE_PLUGIN_DATA}/venv/bin/python" "${CLAUDE_PLUGIN_ROOT}/kb.py" <command>`).

> The agent **message bridge** (cross-agent messaging) needs the external
> `~/.agent-bridge/bridge` binary, which is **not shipped** in this repo. Without
> it the bridge hooks degrade gracefully — you still get findings surfacing and
> kbt; you just don't get cross-agent messaging.

## Skills & commands (plugin)

Installing the plugin adds these skills (invoke as `/<name>` in Claude Code) and one
scaffolding agent. They are **not** standalone CLI commands — they only exist inside a
harness with the plugin loaded.

| Skill / agent | What it does |
|---|---|
| **`/persona`** | List or adopt a session persona. The persona name **is** your bridge identity; adopting one claims the bridge id and loads a binding operating role that the `session-persona` hook re-injects each SessionStart (survives compaction). Personas live at `<project>/.claude/agents/personas/<name>.md`. |
| **`/dispatch <epic-id>`** | Execute an approved **kbt** epic autonomously: claim ready child tasks, spawn implementation agents in waves, verify every diff, close tasks, and commit. |
| **`/kb-setup`** | Guided `kb configure` — embedding provider/model/dim/format/url, summary mode, per-project tracker enablement, `embed-status`, `reembed`. |
| **`/kb-usage`** | How to record/search findings — the type/tag taxonomy, the `--summary` discipline, project scoping, search-first rule. |
| **project-setup** (agent) | Scaffolds a new repo for kb-driven work: `reviewers.yaml`, agent preamble, per-project kb embedding config, and the grep-replacement code-exploration stack. Auto-suggested when `reviewers.yaml` is missing. |

> `/expert-review` is **not** part of this plugin — it's a separate review agent some
> setups install globally. The `/dispatch` workflow expects an already-reviewed epic.

## Configuration

`kb configure` is the supported path. The resolver precedence is **env vars →
`~/.config/kb/config.toml` → defaults**.

`config.toml`:

```toml
[embedding]
url    = "http://localhost:11434/v1/embeddings"
dim    = 1024
format = "openai"          # or "llamacpp"
model  = "qwen3-embedding:0.6b"

[llm]
url          = "http://localhost:8080/completion"   # for --expand + local-llm summaries; unreachable is OK
summary_mode = "extractive"                         # none | extractive | local-llm | subscription-sdk | api
```

Environment overrides (each overrides the toml; a one-line note is logged when it
does). The defaults below point at the original author's dev hosts (`ash`, `tardis`);
`kb configure` writes your own endpoints into `config.toml`, which is the supported path —
you should not need these vars:

| Variable | Description | Default |
|----------|-------------|---------|
| `KB_EMBEDDING_URL` | Embedding endpoint | `http://ash:8081/embedding` |
| `KB_EMBEDDING_DIM` | Embedding dimension | `4096` |
| `KB_EMBEDDING_FORMAT` | `llamacpp` or `openai` | `llamacpp` |
| `KB_EMBEDDING_MODEL` | Model name (for OpenAI-format providers) | (empty) |
| `KB_EMBEDDING_KEY` | API key (secret → `settings.local.json`) | (empty) |
| `KB_LLM_URL` | LLM completion endpoint | `http://tardis:9510/completion` |
| `KB_SUMMARY_MODE` | Summary generation mode | `extractive` |
| `KB_DB` | Database path | `~/.cache/kb/knowledge.db` |
| `KB_SERVER_HOST` | kb-server bind host (configure/unit) | `127.0.0.1` |

Changing the embedding model or dim requires a re-index: `kb reembed --force`
(`kb embed-status` shows configured-vs-stored). `kb configure` health-checks the
embedding endpoint (embeds a probe + verifies the dim) and the LLM endpoint.

## CLI usage

```bash
kb add -t success -p myproject "Fixed the bug by doing X"   # record a finding
kb search "build error"                                     # semantic search (FTS fallback)
kb search --expand "FMHA kernel"                            # LLM query expansion
kb list -n 10 -p myproject                                  # recent findings
kb get <kb-id>                                              # full entry
kb correct <kb-id> "new content" -r "old approach was wrong"
kb surface "topic"                                          # multi-source surface: code symbols + findings + bridge
kb surface --prompt "..." | --file PATH | --all "..."       # preview what the surfacing hooks would inject
kb stats                                                    # counts by type/project
kb embed-status                                             # embedding config vs stored
kb --db /path/to/other.db <command>                         # override the database
```

Run `kb` with no args (or `kb --help`) for the command list. Set `KB_AGENT=0`
for the colorized human-mode help.

### Agent bridge — `kb bridge`

Host-local cross-agent messaging. Peers see each other, send directed messages, and
idle sessions are woken when addressed. Incoming mail auto-injects into your context
each turn, so you rarely need `recv` by hand.

```bash
kb bridge announce <name> --role "..."     # join the bridge under a persona name
kb bridge send <to> "subject" --body "..." # message a peer (body on stdin or --body)
kb bridge send <to> "re: ..." --reply <id> # answer a message that needed a reply
kb bridge recv                             # drain your mailbox
```

> The bridge **transport** (`/bridge/*` on kb-server) ships in this repo, but the
> agent **registry** (announce/whoami) lives in the external `~/.agent-bridge/bridge`
> binary that is **not** bundled. Without it, messaging stays dark while findings
> surfacing and `kbt` keep working.

## Issue tracking — `kbt`

`kbt` is a kb-native, bd-compatible issue tracker. On a host **without** `bd` it
defaults to the self-contained **kb backend** (no external DB); where `bd` is
present it defers to dolt (with a one-line migration notice). Backend resolution,
highest first: `KBT_BACKEND` env → per-project `.kbt/config.toml [tracker] backend`
→ legacy `.beads/config.yaml backend:` (deprecated) → host `~/.config/kb/config.toml
[tracker] backend` → dolt-if-`bd` default. Enable the kb backend for a new project
with `kb configure --project <tag> --enable-tracker` (writes the `.kbt` marker).

To migrate an existing dolt-backed project to kb in one shot:

```bash
kbt bead-migrate            # export dolt → import kb → verify → .kbt marker → archive+remove .beads/
kbt bead-migrate --dry-run  # preview (imports into a throwaway db, verifies, mutates nothing)
kbt bead-migrate --keep-beads   # migrate but leave .beads/ in place
```

It aborts without writing the marker or touching `.beads/` if the export is
truncated or fidelity does not match the live dolt issue count.

`kbt` must be **on PATH** (install step 4) — agents and the lifecycle hooks call
it by name. It needs no live embedding server: issue create/list/close work
offline, and `kbt search` falls back from semantic to FTS when embeddings are
unreachable.

```bash
kbt ready | list | create | show <id> | update <id> | close <id> | dep | blocked
```

## kb-server

A small HTTP/SSE server (`kb serve`, or the systemd `--user` unit from
`kb configure --install-server`) exposing kb/issue read endpoints and the agent
message bridge. It binds **`127.0.0.1` by default** — the endpoints are
unauthenticated, so only bind a non-loopback host (`KB_SERVER_HOST=0.0.0.0`)
behind a trusted network.

| Endpoint | Purpose |
|----------|---------|
| `/kb/search?q=…[&project=…]` | findings search (JSON) |
| `/kb/recent` | recent findings |
| `/kb/finding/{id}` · `/finding/{id}` | one finding |
| `/issues` · `/issues/{id}` | kbt issue reads |
| `/search` · `/` · `/ws` | web UI + websocket |
| `/moim` | **deprecated** — legacy goose ContextProvider feed. Current goose (Phase B adapter) instead calls `/bridge/messages` + `/kb/search` directly and gets zero traffic here; the route is retained only until goose's config marker is repointed (kb-a7694e). |
| `/bridge/agents` | live agent registry |
| `/bridge/messages` · `/bridge/send` | mailbox read / send |
| `/bridge/watch[?since=N]` (SSE) | live message stream |

> These bridge **endpoints** ship, but the agent **registry** (announce/whoami,
> which agents exist) lives in the external `~/.agent-bridge/bridge` binary that
> is **not** in this repo (see the plugin section above). On a fresh host you get
> findings surfacing + `kbt`; cross-agent messaging stays dark until that binary
> is present.

## Database

- Default: `~/.cache/kb/knowledge.db` (override with `--db` or `KB_DB`).
- The DB is gitignored — no findings ship with the repo.
