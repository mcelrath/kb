# Knowledge Base (kb)

SQLite + sqlite-vec powered findings database for tracking successes, failures, experiments, and discoveries across projects.

## Features

- **Vector similarity search** using sqlite-vec for semantic retrieval
- **Full-text search** fallback via SQLite FTS5
- **LLM query expansion** for improved recall (optional)
- **Supersession chains** for correcting outdated findings
- **Project/sprint tagging** for organization
- **MCP server** for Claude Code integration
- **Notation tracking** for project-specific terminology
- **Error pattern database** for build error solutions

## Installation

### Prerequisites

- Python 3.13+
- sqlite-vec Python package
- Access to embedding server (or local sentence-transformers)

### Setup

```bash
cd ~/Projects/ai/kb

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install sqlite-vec

# Optional: for local embeddings without remote server
pip install sentence-transformers
```

### CLI Wrapper

Create `~/.local/bin/kb`:

```bash
#!/bin/bash
exec /home/mcelrath/Projects/ai/kb/.venv/bin/python /home/mcelrath/Projects/ai/kb/kb.py "$@"
```

```bash
chmod +x ~/.local/bin/kb
```

## Claude Code Plugin Install

The recommended way to use kb with Claude Code is as a plugin. This requires no
entries in your `~/.claude/CLAUDE.md` — the plugin auto-injects kb conventions
and surfacing on every SessionStart.

### Install

```bash
# 1. Add the kb marketplace (once per machine)
claude plugin marketplace add /path/to/kb

# 2. Install the plugin
claude plugin install kb@kb-local
```

Replace `/path/to/kb` with the cloned repo root (the directory containing
`.claude-plugin/`). To install from a URL, pass the GitHub repo URL to
`marketplace add`.

### What happens on SessionStart

1. **setup-venv.sh** — builds a deps-only venv at `$CLAUDE_PLUGIN_DATA/venv`
   (idempotent, hash-gated; requires internet access to pypi.org on first run).
2. **env-probe.sh** — confirms `CLAUDE_PLUGIN_ROOT` and `CLAUDE_PLUGIN_DATA`
   are exported and injects them as context.
3. **kb-flush-pending.sh** — drains any queued `~/.claude/pending-kb-adds/` entries.
4. **kb-context.sh** — injects kb conventions (search-first, --summary discipline,
   types/tags taxonomy) into the session context. If the embedding server is
   unreachable, it emits a `KB-INFRA DOWN` warning and falls back to FTS-only mode.
5. **scaffold-check.sh** — detects missing `reviewers.yaml` / `agent-preamble.md`
   and prompts to run the `project-setup` agent.

### Using kb and kbt from the plugin

After SessionStart, the plugin venv is at `$CLAUDE_PLUGIN_DATA/venv`. The hooks
invoke kb as:

```bash
"${CLAUDE_PLUGIN_DATA}/venv/bin/python" "${CLAUDE_PLUGIN_ROOT}/kb.py" <command>
```

The `kbt` script in the plugin root is callable the same way. For interactive use,
create a shell alias or wrapper pointing at those paths, or run `kb configure` to
set up a `~/.local/bin/kb` wrapper.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KB_EMBEDDING_URL` | Remote embedding endpoint | (empty, uses local model) |
| `KB_EMBEDDING_DIM` | Embedding dimension | 4096 |
| `KB_LLM_URL` | LLM completion endpoint for query expansion | http://tardis:9510/completion |

### Machine-Specific Configuration

**tardis** (local development):
```bash
export KB_EMBEDDING_URL="http://ash:8080/embedding"
export KB_EMBEDDING_DIM=4096
export KB_LLM_URL="http://tardis:9510/completion"
```

**ash** (GPU server):
```bash
export KB_EMBEDDING_URL="http://localhost:8080/embedding"
export KB_EMBEDDING_DIM=4096
export KB_LLM_URL="http://localhost:9510/completion"
```

## Claude Code Integration

### MCP Server Setup

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/home/mcelrath/Projects/ai/kb/.venv/bin/python",
      "args": ["/home/mcelrath/Projects/ai/kb/kb_mcp.py"],
      "env": {
        "KB_EMBEDDING_URL": "http://ash:8080/embedding",
        "KB_EMBEDDING_DIM": "4096",
        "KB_LLM_URL": "http://tardis:9510/completion"
      }
    }
  }
}
```

### Permissions

Add to `~/.claude/settings.json` in the `permissions.allow` array:

```json
"mcp__knowledge-base__kb_add",
"mcp__knowledge-base__kb_search",
"mcp__knowledge-base__kb_correct",
"mcp__knowledge-base__kb_list",
"mcp__knowledge-base__kb_get",
"mcp__knowledge-base__kb_stats",
"mcp__knowledge-base__kb_doc_add",
"mcp__knowledge-base__kb_doc_citations",
"mcp__knowledge-base__kb_doc_cite",
"mcp__knowledge-base__kb_doc_finding_docs",
"mcp__knowledge-base__kb_doc_get",
"mcp__knowledge-base__kb_doc_list",
"mcp__knowledge-base__kb_doc_search",
"mcp__knowledge-base__kb_doc_supersede",
"mcp__knowledge-base__kb_error_add",
"mcp__knowledge-base__kb_error_get",
"mcp__knowledge-base__kb_error_link",
"mcp__knowledge-base__kb_error_list",
"mcp__knowledge-base__kb_error_search",
"mcp__knowledge-base__kb_error_solutions",
"mcp__knowledge-base__kb_error_verify",
"mcp__knowledge-base__kb_notation_add",
"mcp__knowledge-base__kb_notation_history",
"mcp__knowledge-base__kb_notation_list",
"mcp__knowledge-base__kb_notation_search",
"mcp__knowledge-base__kb_notation_update"
```

## CLI Usage

```bash
# Add a finding
kb add --type success --project myproject "Fixed the bug by doing X"

# Search findings
kb search "build error"

# Search with query expansion (uses LLM)
kb search --expand "FMHA kernel"

# Search with verbose output (shows expanded query)
kb -v search --expand "quaternion"

# List recent findings
kb list --limit 10

# Correct a finding
kb correct <finding-id> --reason "Previous approach was wrong" "New correct approach"

# Show statistics
kb stats
```

## Database Location

- Default: `~/.cache/kb/knowledge.db`
- Override with `--db-path` flag

## Query Expansion

When `--expand` is used, the query is sent to an LLM to add related terms:

```
Original: "FMHA kernel"
Expanded: "FMHA kernel FlashMultiAttention FP8 Transformer inference self-attention..."
```

This improves recall by including synonyms, acronym expansions, and related concepts.

Expansions are cached in-memory for the duration of the process.
