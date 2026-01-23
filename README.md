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
