---
name: kb-usage
description: How to use the knowledge base (kb) — record a finding, search prior findings, the type/tag taxonomy, the --summary discipline, project scoping, and the search-first rule. Invoke when asked how to add/search kb, what tags or types to use, or why a kb add was rejected.
---

# Using the knowledge base (kb)

The kb is a vector + FTS store of findings, shared across sessions and (via the plugin) across coding harnesses. **Search before you add. Add with a summary.**

## Core commands

```
kb search "query"                 # semantic + FTS; FIRST search unfiltered (no -p)
kb search "query" -p PROJECT -t TYPE -n 10
kb add "content" -t TYPE -p PROJECT --tags T1,T2 --summary "<one dense sentence>"
kb get <kb-id>                    # full entry
kb list -p PROJECT -t TYPE        # browse
kb correct <kb-id> "new content" -r "reason"   # supersede an outdated finding
kb related <kb-id>                # semantically similar findings
```

## The two non-negotiable rules

1. **Search first, then add.** Run `kb search "topic"` (unfiltered first, then narrow by `-p`) before recording — avoids duplicates and surfaces prior art. If a finding exists and is wrong, `kb correct` it rather than adding a contradictory one.
2. **Always pass `--summary "<one sentence>"`.** You wrote the finding, so you write its one-line summary in the same turn — it's free and far better than the no-LLM extractive fallback. The summary is what shows in search results: make it dense and specific (key result + identifiers), not a restatement of the first sentence.

## Taxonomy

- **Type** (`-t`): `success | failure | experiment | discovery | correction`
- **Tags** (`--tags`), two axes:
  - confidence: `proven | heuristic | open-problem`
  - importance: `core-result | technique | detail`
  - plus free topic tags (e.g. `lsp,chunker`).

## Project scoping

`-p PROJECT` scopes the finding. Use the canonical project tag for the repo. The **first** search of any investigation should be unfiltered (cross-project blindness is a real failure mode); then narrow with `-p`.

## When kb infra is down

If the embedding server is unreachable, search/surfacing is BLIND (empty ≠ "nothing found") and `kb add` queues to a pending file. Fallback for adds: write `~/.claude/pending-kb-adds/<UTC>-<session>.txt` with a `# type:` / `# project:` / `# tags:` header; `kb flush-pending` drains it on recovery. Never fall back to a stray `.md` file.
