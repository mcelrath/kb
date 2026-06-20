# Agent Preamble — Knowledge Base (kb)

Read this BEFORE starting your task. This is a condensed agent-targeted summary of the critical rules — read it first even though CLAUDE.md is also in your context.

## The Project

`kb` is a SQLite + sqlite-vec powered findings database for tracking successes, failures,
experiments, and discoveries across AI/physics projects. It exposes a Python library
(`kb/`) and a CLI (`kb`, on PATH) consumed by Claude Code and its hooks.

## Non-Negotiable Constraints

- Methods return dicts, not scalars: `add()` returns `{"id": "...", "tags_suggested": True, ...}` — never a bare string
- Script registry key is `"filename"` not `"file"` — the wrong key causes `KeyError: 'file'` at runtime
- Embedding cache: module-level LRU with 500-entry cap; eviction must be verified when touching `kb/core/embedding.py`
- Similarity formula (L2-normalized vectors): `similarity = 1 - (distance ** 2) / 2` — any other formula is wrong
- No dead code, no backward-compat wrappers, no mocks — DELETE superseded code
- Database location: `~/.cache/kb/knowledge.db`; embedding server: `KB_EMBEDDING_URL` (default http://ash:8080/embedding)
- No `grep`/`rg` on source files — hook blocks it; use `ast-grep` or `Read` whole files

## Architecture

| Module | Purpose |
|--------|---------|
| `kb/core/schema.py` | DDL, table definitions, migration |
| `kb/core/connection.py` | SQLite connection, WAL mode, retry |
| `kb/core/embedding.py` | Remote embedding calls + LRU cache |
| `kb/facade.py` | Public KnowledgeBase class; delegates to entity repos |
| `kb/entities/base.py` | BaseRepository; raw SQL in entity repos, not facade |
| `kb/entities/findings.py` | Core findings CRUD + supersession chains |
| `kb/entities/theorems.py` | Lean theorem entries, finding_id, drift detection |
| `kb/entities/documents.py` | Document citations, doc-finding links |
| `kb/entities/scripts.py` | Script registry (key: "filename") |
| `kb/entities/concepts.py` | Notation/concept tracking |
| `kb/search/hybrid.py` | FTS5 + vector hybrid search, BM25 + cosine fusion |
| `kb/llm/client.py` | LLM completion client (tardis:9510), thinking=false |
| `kb/llm/analysis.py` | LLM-based summarization, query expansion |
| `kb/hooks/` | Claude Code PreToolUse/PostToolUse hook implementations |
| `kb.py` | CLI entry point; subcommands mirror facade methods |
| `kb/issue_cli.py` | `kbt` issue-tracker CLI (kb-native, bd-compatible) |
| `scripts/ingest_lean_direct.py` | Lean theorem ingestion |
| `scripts/ingest_python.py` | Python symbol ingestion |
| `scripts/ingest_tex.py` | LaTeX annotation ingestion |

## Key Patterns

- `KnowledgeBase` in `kb/facade.py` is the single public API; CLI and MCP both go through it
- Entity repos in `kb/entities/` own all SQL; facade only calls repo methods
- `TheoremRepository.add()` does name-based reconciliation (upsert by name, not duplicate insert)
- Lean-verify subcommand checks theorem drift between KB and current Lean source

## Common Bugs (Do NOT Reintroduce)

| Bug | Location | Correct Fix |
|-----|----------|-------------|
| `KeyError: 'file'` | script dict consumers | use `r['filename']` |
| Embedding cache unbounded growth | `kb/core/embedding.py` | evict at 500 entries |
| `similarity = 1 - distance` | search results | use `1 - (d**2)/2` for L2-norm |
| Theorem duplicate insert | `TheoremRepository.add()` | upsert by name |
| `HTTP 500` from embedding | `kb/core/embedding.py` | retry with exponential backoff (max_retries=5) |
| `RemoteDisconnected` LLM | `kb/llm/client.py` | retry; check tardis:9510 |
| `format()` on Lean/LaTeX curly braces | `ingest_lean_direct.py` | escape `{` as `{{` before format |

## Anti-Patterns

| If you write... | Stop because... |
|-----------------|-----------------|
| `rg`/`grep` on `.py`/`.lean` files | Hook blocks it; use `ast-grep` or `Read` |
| New `.md` files (except allowlisted) | Hook blocks it; route to `kb add` or inline |
| `return entry_id` from a mutating method | Must return dict with id and metadata |
| `r['file']` on a script result dict | KeyError; must be `r['filename']` |
| SQL in `kb/facade.py` | Architecture violation; SQL belongs in entity repos |
| Duplicate theorem insert on re-ingest | Use upsert-by-name in `TheoremRepository` |
| Embedding server assumed always available | Add retry + graceful degradation |
| Hardcoding `~/.local/bin/kb` | `kb`/`kbt` are on PATH; call them by name |

## Epistemological Rules

1. "Not Found" means "I searched and found no evidence." Not "it doesn't exist."
2. Code > Comments > KB > assumptions.
3. 5 rounds of kb-research minimum, not 2.
4. Verify, don't infer. Read files; don't describe from grep output.
5. State evidence. Every claim cites file:line, kb-ID, or command output.
6. `kb add` before returning. Checkpoint every 10 tool uses.
7. Use `project="knowledge-base"` for all `kb add`/`kb search` calls.

## Tracking: kbt (durable) vs the harness task-list (ephemeral)

Two tools, two scopes — NEVER parallel trackers. **kbt** (`kbt create/ready/close`, kb-native,
bd-compatible) is the DURABLE source of truth: epics/tasks/deps/priorities that survive the
session. The **harness task-list (TaskCreate/TodoWrite)** is a session-local execution view —
the step checklist for the ONE kbt task you are executing now; it dies with the session, so
nothing may exist only there. The harness "consider TaskCreate" reminder is advisory: use a
task-list only for a claimed kbt task with ≥3 sequential steps, mirroring that task's steps
(not other kbt tasks, not a single edit/dispatch). Dispatching an agent is NOT a todo — it
serves a kbt task. Finish → tick todos → `kbt close` (mandatory) → commit.

## Stopping Conditions

Stop and return partial results if:
- Same error 3 times consecutively
- 10+ tool calls with no new findings
- 5+ search phrasings with no results
- 8+ files read without concrete output
