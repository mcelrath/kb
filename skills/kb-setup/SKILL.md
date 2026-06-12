---
name: kb-setup
description: How to configure kb for a host or project — kb configure (embedding provider/model/dim/format/url, summary mode), per-project tracker enablement, the secret-handling guard, embed-status, and reembed after a model change. Invoke when setting up kb on a new host, enabling kb for a project, or changing the embedding model.
---

# Configuring kb

Two layers: a **global** host config (once) and a **per-project** config (per repo).

## Global (once per host)

```
kb configure --provider ollama --model qwen3-embedding:0.6b --dim 1024 \
  --format openai --url http://localhost:11434/v1/embeddings --summary-mode extractive
```

Writes non-secret `KB_*` env to `settings.json` (MERGE, never clobber). A secret
`KB_EMBEDDING_KEY` goes to `settings.local.json` ONLY after `git check-ignore`
confirms it is ignored (refuses otherwise). Seeds the `embedding_meta` table.

Env vars: `KB_EMBEDDING_FORMAT` (`llamacpp`|`openai`), `KB_EMBEDDING_URL`,
`KB_EMBEDDING_MODEL`, `KB_EMBEDDING_DIM`, `KB_EMBEDDING_KEY` (secret),
`KB_SUMMARY_MODE` (`none`|`extractive`|`local-llm`|`subscription-sdk`|`api`), `KB_DB`.
Unset ⇒ defaults to ash:8081 llamacpp 4096.

### Embedding model choice (code + science)

- **CPU / no GPU (default):** `ollama pull qwen3-embedding:0.6b` (1024d, MTEB-Code 75, within timeout).
- **Free hosted, code-specialized:** voyage-code-3 (1024d, needs API key, zero local compute).
- **GPU (~16GB):** `qwen3-embedding:8b` (4096d, top quality). Do NOT run the 8B on CPU.
- AVOID nomic-embed-text / all-MiniLM as default (weak on code/science). Target dim = 1024.

## Per-project (non-interactive, agent-safe)

```
kb configure --project <tag> --project-dir <root>   # + --enable-tracker if the project uses kbt
```

Reuses the global embedding config; sets a per-project `KB_DB` and (with
`--enable-tracker`) the `.beads/config.yaml: backend: kb` flag.

## Status + reembed

```
kb embed-status        # configured-vs-stored model/dim + verdict
kb reembed --force     # REQUIRED after any model/dim change — a dim change DROPs+recreates
                       # all _vec tables at the new dim and re-embeds (FTS covers the window)
```

## Caution

`kb configure` WITHOUT `--config-dir` defaults to the real `~/.claude` — tests and
smokes MUST pass `--config-dir <temp>` or they clobber the live config.
