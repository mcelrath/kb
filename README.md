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
- **Claude Code plugin** — findings surfacing + conventions injection on every
  SessionStart, no `CLAUDE.md` edits or permission all-listing required.
- **`kbt` issue tracker** — kb-native, bd-compatible CLI; no external DB on a
  fresh host.
- **`kb-server`** — HTTP/SSE endpoints (kb/issue reads + an agent message bridge).

There is **no MCP server** — all kb operations go through the `kb` CLI (and, inside
Claude Code, the plugin hooks). References to `kb_mcp.py` or `mcp__knowledge-base__*`
tools are stale.

## Prerequisites

- **Python 3.14+** (3.11–3.13 may fail to import due to builtin-shadowing in class
  bodies; portability fix tracked).
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
uv venv --python python3.14 --seed
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

# 4. Put `kb` AND `kbt` (the issue tracker) on PATH. Agents and the plugin's
#    git/lifecycle hooks invoke `kbt` BY NAME — without it on PATH, issue
#    tracking falls back to nothing on a host that also lacks `bd`.
mkdir -p ~/.local/bin
printf '#!/bin/bash\nexec %s %s "$@"\n' "$PWD/.venv/bin/python" "$PWD/kb.py" > ~/.local/bin/kb
printf '#!/bin/bash\nexec %s %s "$@"\n' "$PWD/.venv/bin/python" "$PWD/kbt"   > ~/.local/bin/kbt
chmod +x ~/.local/bin/kb ~/.local/bin/kbt
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo 'add ~/.local/bin to PATH' ;; esac
```

`kb configure` writes `~/.config/kb/config.toml` (the source of truth) and mirrors
the non-secret values into the config dir's `settings.json`; any API key goes to
`settings.local.json` (only after `git check-ignore` confirms it's ignored).

## Claude Code plugin

The recommended way to use kb with Claude Code. The plugin's hooks inject kb
conventions and surface findings on every SessionStart — no `CLAUDE.md` entries.

```bash
# Add the local marketplace (the repo root, containing .claude-plugin/) + install
claude plugin marketplace add /path/to/kb
claude plugin install kb@kb-local
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

### What happens on SessionStart

1. **setup-venv.sh** — builds a deps-only venv at `$CLAUDE_PLUGIN_DATA/venv`
   (idempotent, hash-gated; needs pypi.org access on first run).
2. **env-probe.sh** — confirms/injects `CLAUDE_PLUGIN_ROOT` / `CLAUDE_PLUGIN_DATA`.
3. **kb-flush-pending.sh** — drains any queued offline kb-adds.
4. **kb-context.sh** — injects kb conventions; emits a `KB-INFRA DOWN` warning and
   falls back to FTS-only if the embedding server is unreachable.
5. **scaffold-check.sh** — flags a missing `reviewers.yaml` and offers project setup.

Inside Claude Code the hooks invoke kb as
`"${CLAUDE_PLUGIN_DATA}/venv/bin/python" "${CLAUDE_PLUGIN_ROOT}/kb.py" <command>`.

> The agent **message bridge** (cross-agent messaging) needs the external
> `~/.agent-bridge/bridge` binary, which is **not shipped** in this repo. Without
> it the bridge hooks degrade gracefully — you still get findings surfacing and
> kbt; you just don't get cross-agent messaging.

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

Environment overrides (each overrides the toml; a one-line note is logged when it does):

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
kb stats                                                    # counts by type/project
kb embed-status                                             # embedding config vs stored
kb --db /path/to/other.db <command>                         # override the database
```

Run `kb` with no args (or `kb --help`) for the full command list. Set `KB_AGENT=0`
for the colorized human-mode help.

## Issue tracking — `kbt`

`kbt` is a kb-native, bd-compatible issue tracker. On a host **without** `bd` it
defaults to the self-contained **kb backend** (no external DB); where `bd` is
present it defers to dolt, and an explicit `.beads/config.yaml` `backend:` always
wins. Enable the kb backend for a project with
`kb configure --project <tag> --enable-tracker`.

`kbt` must be **on PATH** (install step 4) — agents and the lifecycle hooks call
it by name. It needs no live embedding server: issue create/list/close work
offline, and `kbt search` falls back from semantic to FTS when embeddings are
unreachable.

```bash
kbt ready | list | create | show <id> | update <id> | close <id> | dep | blocked
```

## kb-server

A small HTTP/SSE server (`kb serve`, or the systemd `--user` unit from
`kb configure --install-server`) exposing kb/issue read endpoints and an agent
message bridge (`/bridge/messages`, `/bridge/watch` SSE, `/bridge/send`). It binds
**`127.0.0.1` by default** — the endpoints are unauthenticated, so only bind a
non-loopback host (`KB_SERVER_HOST=0.0.0.0`) behind a trusted network.

> These bridge **endpoints** ship, but the agent **registry** (announce/whoami,
> which agents exist) lives in the external `~/.agent-bridge/bridge` binary that
> is **not** in this repo (see the plugin section above). On a fresh host you get
> findings surfacing + `kbt`; cross-agent messaging stays dark until that binary
> is present.

## Database

- Default: `~/.cache/kb/knowledge.db` (override with `--db` or `KB_DB`).
- The DB is gitignored — no findings ship with the repo.
