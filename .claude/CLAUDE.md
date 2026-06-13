project: knowledge-base

# Knowledge Base Project - Development Guide

## Architecture

```
kb.py              # CLI entry: argparse + dispatch table -> kb/cli/commands/
kb/cli/commands/   # one module per command group (findings/admin/maintenance/ingest/lean/serve/misc)
kb/facade.py       # KnowledgeBase facade (the library API)
kb/core/           # connection, schema, embedding
kb/entities/       # repositories (scripts/documents/theorems/concepts/issues/bridge)
kb/search/         # hybrid (vector + FTS, RRF)
kb/server/         # kb serve: create_app factory + routes/bridge/live/renderers + templates
kb/config.py       # KbConfig + load_config (env -> ~/.config/kb/config.toml -> defaults)
kb/ingest/         # ingest entry points (imported as library by kb ingest)
kb/markdown.py     # dependency-free markdown helpers (CLI + web)
curate_kb.py       # Automated curation: tagging, consolidation, entry points
```

(There is NO MCP server: the dual kb_mcp_core.py/kb_mcp_advanced.py were dead
duplication, deleted in kb-ez9.7. All kb ops go through the `kb` CLI.)

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

- GLOBAL (once per host, interactive or flags): `kb configure --provider ollama --model qwen3-embedding:0.6b --dim 1024 --format openai --url http://localhost:11434/v1/embeddings --summary-mode extractive`. Writes non-secret `KB_*` env to `settings.json` (MERGE, never clobber); `KB_EMBEDDING_KEY` → `settings.local.json` ONLY after `git check-ignore` confirms it's ignored (refuses otherwise). Seeds `embedding_meta`.
- PER-PROJECT (non-interactive, agent-safe): `kb configure --project <tag> --enable-tracker [--db PATH]`. Writes `.beads/config.yaml: backend: kb` (merge) + per-project `KB_DB`. Reuses global. `/project-setup` runs this as its Phase-5 tail.

Env vars: `KB_EMBEDDING_FORMAT` (`llamacpp`|`openai`), `KB_EMBEDDING_URL`, `KB_EMBEDDING_MODEL`, `KB_EMBEDDING_DIM`, `KB_EMBEDDING_KEY` (secret→settings.local.json), `KB_SUMMARY_MODE` (`none`|`local-llm`|`subscription-sdk`|`api`), `KB_DB`. Unset ⇒ defaults to ash:8081 llamacpp 4096.

Embedding-model identity is tracked in `embedding_meta`. `kb embed-status` shows configured-vs-stored + verdict. CHANGING model/dim requires `kb reembed --force` (a dim change DROPs+recreates all 7 `_vec` tables at the new dim and re-embeds; FTS covers the window). Embedding model choice (code+science benchmarks, June 2026 research — kb-au2):
- **CPU / no GPU (recommended default):** `ollama pull qwen3-embedding:0.6b` (1024d, Apache-2.0, MTEB-Code 75 — ~0.2-0.6s/embed on CPU, within timeout).
- **Free hosted, code-specialized:** voyage-code-3 (1024d, 200M tokens free, needs API key) — zero local compute.
- **GPU (~16GB):** `qwen3-embedding:8b` (4096d, top quality). Do NOT run an 8B embedder on CPU (1-6s/embed — blows the timeout).
- AVOID nomic-embed-text / all-MiniLM as the default: confirmed WEAK on code+science (~15-25pt MTEB retrieval gap, absent from code leaderboards). Vec-table dim target = 1024. `KB_EMBED_TIMEOUT` default 180s is a generous ceiling; the search path fast-fails to FTS.

Summaries: the CALLER provides one — `kb add --summary "<one sentence>"`. The agent that wrote the finding writes its summary in the same turn (free, smart, no second model); this is preferred and is the documented convention. `KB_SUMMARY_MODE` is only the FALLBACK when no `--summary` is passed: `extractive` (default, no-LLM first-sentence) | `none` | `local-llm` (tardis:9510) | `subscription-sdk` (Haiku via claude_agent_sdk, pending Agent-SDK credits, kb-zi9) | `api`.

CAUTION: `kb configure` WITHOUT `--config-dir` defaults to the real `~/.claude` — tests/smokes MUST pass `--config-dir <temp>` or they clobber the live config (see kb-2c3 retro).

## Adding New Features

1. Add method to the `KnowledgeBase` facade in `kb/facade.py` (delegating to a `kb/entities/` repository if it owns SQL)
2. Add an argparse subcommand in `kb.py main()` + a `run(kb, args)` handler in `kb/cli/commands/<group>.py`, and register it in the `main()` dispatch table
3. If it needs config, source it via `kb/config.py` `load_config()` (never read env at import)
