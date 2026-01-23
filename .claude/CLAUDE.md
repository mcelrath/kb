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
- Embeddings: remote server at `KB_EMBEDDING_URL` (default: http://ash:8080/embedding)
- LLM: remote server at `KB_LLM_URL` (default: http://tardis:9510/completion)

## Running

```bash
# CLI
KB_EMBEDDING_URL=http://ash:8080/embedding KB_EMBEDDING_DIM=4096 \
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
| `HTTP 500 from embedding` | Server overloaded | Retry or check ash:8080 |
| `RemoteDisconnected` | LLM server timeout | Check tardis:9510 |
| Duplicate findings | Not searching first | Use `kb_detect_duplicates` before add |

## Adding New Features

1. Add method to `KnowledgeBase` class in kb.py
2. Add CLI subcommand in `main()`
3. Add MCP tool in kb_mcp.py with `@mcp.tool()` decorator
4. Add to settings.json permissions if needed
