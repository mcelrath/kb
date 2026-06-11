project: knowledge-base

# Knowledge Base Project - Development Guide

## Architecture

```
kb.py          # Core library: KnowledgeBase class, CLI
kb_mcp.py      # MCP server: exposes kb.py as MCP tools
curate_kb.py   # Automated curation: tagging, consolidation, entry points
```

## Database

- Location: `~/.cache/kb/knowledge.db`
- SQLite + sqlite-vec for vector similarity
- Embeddings: remote server at `KB_EMBEDDING_URL` (default: http://ash:8081/embedding)
- LLM: remote server at `KB_LLM_URL` (default: http://tardis:9510/completion)

## Running

```bash
# CLI
KB_EMBEDDING_URL=http://ash:8081/embedding KB_EMBEDDING_DIM=4096 \
  .venv/bin/python kb.py <command>

# MCP server (started automatically by Claude Code)
# Config in ~/.claude/settings.json under mcpServers
```

## Key Patterns

### Return Dicts, Not Scalars
Methods return dicts with metadata for MCP tools to format:
```python
def add(...) -> dict:  # Returns {"id": "...", "tags_suggested": True, ...}
def error_add(...) -> dict:  # Returns {"id": "...", "is_new": True, ...}
```

### Script Registry Key Name
Scripts use `"filename"` not `"file"` in returned dicts:
```python
{"id": "...", "filename": "script.py", "purpose": "..."}
```

### Embedding Cache
Module-level LRU cache for embeddings (500 entries max):
```python
_embedding_cache: dict[str, list[float]]
_embedding_cache_order: list[str]
```

### Similarity Formula
For L2-normalized vectors, cosine similarity from L2 distance:
```python
similarity = 1 - (distance ** 2) / 2
```

## Testing

```bash
# Syntax check
python3 -m py_compile kb.py kb_mcp.py

# Run CLI
.venv/bin/python kb.py stats
.venv/bin/python kb.py search "test query"

# Test MCP tools (via Claude Code)
# Use kb_stats(), kb_search(), etc. directly
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'file'` | Script dict key mismatch | Use `r['filename']` not `r['file']` |
| `HTTP 500 from embedding` | Server overloaded | Retry or check ash:8081 |
| `RemoteDisconnected` | LLM server timeout | Check tardis:9510 |
| Duplicate findings | Not searching first | Use `kb_detect_duplicates` before add |

## KB Configuration (embedding + summaries) — epic kb-2c3

Two-layer config via `kb configure`:

- GLOBAL (once per host, interactive or flags): `kb configure --provider ollama --model qwen3-embedding:8b --dim 4096 --format openai --url http://localhost:11434/v1/embeddings --summary-mode local-llm`. Writes non-secret `KB_*` env to `settings.json` (MERGE, never clobber); `KB_EMBEDDING_KEY` → `settings.local.json` ONLY after `git check-ignore` confirms it's ignored (refuses otherwise). Seeds `embedding_meta`.
- PER-PROJECT (non-interactive, agent-safe): `kb configure --project <tag> --enable-tracker [--db PATH]`. Writes `.beads/config.yaml: backend: kb` (merge) + per-project `KB_DB`. Reuses global. `/project-setup` runs this as its Phase-5 tail.

Env vars: `KB_EMBEDDING_FORMAT` (`llamacpp`|`openai`), `KB_EMBEDDING_URL`, `KB_EMBEDDING_MODEL`, `KB_EMBEDDING_DIM`, `KB_EMBEDDING_KEY` (secret→settings.local.json), `KB_SUMMARY_MODE` (`none`|`local-llm`|`subscription-sdk`|`api`), `KB_DB`. Unset ⇒ defaults to ash:8081 llamacpp 4096.

Embedding-model identity is tracked in `embedding_meta`. `kb embed-status` shows configured-vs-stored + verdict. CHANGING model/dim requires `kb reembed --force` (a dim change DROPs+recreates all 7 `_vec` tables at the new dim and re-embeds; FTS covers the window). Recommended local default: `ollama pull qwen3-embedding:8b` (4096 — matches the ash index, NO re-embed) or `nomic-embed-text` (768, lighter, needs re-embed).

Summaries default to `local-llm` (tardis:9510). `subscription-sdk` (Haiku via claude_agent_sdk, scrubs a stale `ANTHROPIC_API_KEY`) is built but pending an Agent-SDK credit pool (kb-zi9).

CAUTION: `kb configure` WITHOUT `--config-dir` defaults to the real `~/.claude` — tests/smokes MUST pass `--config-dir <temp>` or they clobber the live config (see kb-2c3 retro).

## Adding New Features

1. Add method to `KnowledgeBase` class in kb.py
2. Add CLI subcommand in `main()`
3. Add MCP tool in kb_mcp.py with `@mcp.tool()` decorator
4. Add to settings.json permissions if needed
