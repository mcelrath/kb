#!/bin/bash

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# --- EMBEDDING-DOWN gate (ash:8081): surface hard STOP instead of blind retrieval ---
# SessionStart injects this hook's STDOUT into the agent's context (stderr is only
# logged), so the embedding-down warning MUST go to stdout to actually reach the
# agent at session start (kb-zma part 1). It is emitted FIRST so it leads the
# context block.
. "${PLUGIN_ROOT}/hooks/scripts/lib/ash_health.sh" 2>/dev/null || true
if command -v ash_down >/dev/null 2>&1 && ash_down; then
  echo "$ASH_STOP_LINE"
fi

# KB Context Injection Hook
# Shows recent findings for current project.
# TTY-aware context (handoff.md path deprecated and stripped).
source "${PLUGIN_ROOT}/hooks/scripts/lib/claude-env.sh"

# Resolve venv python from venv-path.sh
source "${PLUGIN_ROOT}/hooks/scripts/lib/venv-path.sh"
KB_VENV_PYTHON="${KB_VENV_PYTHON:-${KB_VENV_DIR}/bin/python}"
KB_SCRIPT="${PLUGIN_ROOT}/kb.py"

# Gracefully exit if KB tools not installed
[[ ! -f "$KB_SCRIPT" || ! -x "$KB_VENV_PYTHON" ]] && exit 0
CONTEXT_FILE="$HOME/.cache/kb/last_work_context.txt"

# Get project name from git root or current directory
if git rev-parse --show-toplevel &>/dev/null; then
    PROJECT=$(basename "$(git rev-parse --show-toplevel)")
else
    PROJECT=$(basename "$PWD")
fi

# Skip if no project detected
if [[ -z "$PROJECT" ]]; then
    exit 0
fi

export KB_EMBEDDING_URL="${KB_EMBEDDING_URL:-http://ash:8081/embedding}"
export KB_EMBEDDING_DIM=4096

# Show last work context if available and recent (within last hour)
if [[ -f "$CONTEXT_FILE" ]]; then
    CONTEXT_AGE=$(($(date +%s) - $(python3 -c "import os;print(int(os.path.getmtime('$CONTEXT_FILE')))" 2>/dev/null || echo 0)))
    if [[ $CONTEXT_AGE -lt 3600 ]]; then
        SAVED_PROJECT=$(grep "^Project:" "$CONTEXT_FILE" | cut -d: -f2- | xargs)
        if [[ "$SAVED_PROJECT" == "$PROJECT" ]]; then
            echo "=== Last Session Context ==="
            grep "^Context:" "$CONTEXT_FILE" | cut -d: -f2-
            echo ""
        fi
    fi
fi

# Surface findings: SEMANTIC (relevance) when the saved session context gives a
# query signal; else fall back to recency. (kb-mrl: recency -> vector-query.)
# Per-prompt semantic surfacing lives in kb-prompt-surface.py (UserPromptSubmit);
# this SessionStart path only has the resume context to query with.
KB_IDS=""
CTX_QUERY=$(grep "^Context:" "$CONTEXT_FILE" 2>/dev/null | cut -d: -f2- | tr '\n' ' ' | head -c 300)
if [[ -n "$CTX_QUERY" ]]; then
    # Bound the SessionStart semantic search: a slow embedding server must not
    # block session start. Short single-attempt embed timeout (slow -> FTS) + a
    # hard outer cap; degrades to the recency fallback below if it can't answer.
    HITS=$(KB_SEARCH_EMBED_TIMEOUT=8 KB_SEARCH_EMBED_RETRIES=0 timeout 12 \
        "$KB_VENV_PYTHON" "$KB_SCRIPT" search "$CTX_QUERY" --project="$PROJECT" --limit=5 --json 2>/dev/null) || true
    if [[ -n "$HITS" ]]; then
        KB_IDS=$(printf '%s' "$HITS" | "$KB_VENV_PYTHON" -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: d=[]
print(' '.join(r['id'] for r in d if float(r.get('similarity') or 0) >= 0.42)[:120])" 2>/dev/null)
    fi
    [[ -n "$KB_IDS" ]] && echo "Relevant KB ($PROJECT, semantic): $KB_IDS"
fi

# Recency fallback when no semantic query/hits available.
if [[ -z "$KB_IDS" ]]; then
    FINDINGS=$("$KB_VENV_PYTHON" "$KB_SCRIPT" list --project="$PROJECT" --limit=3 2>/dev/null) || true
    if [[ -n "$FINDINGS" && "$FINDINGS" != "No findings found." ]]; then
        KB_IDS=$(echo "$FINDINGS" | grep -oE 'kb-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}' | head -5 | tr '\n' ' ')
        [[ -n "$KB_IDS" ]] && echo "Recent KB ($PROJECT): $KB_IDS"
    fi
fi

# --- KB CONVENTIONS (injected every session so CLAUDE.md prose is not needed) ---
cat <<'CONVENTIONS'

=== KB Conventions (kb plugin) ===
SEARCH FIRST then ADD. kb search "topic" (no -p first); then narrow by project.
ALWAYS pass --summary "<one sentence>" to kb add — you wrote the finding, write the summary.
  ~/.local/bin/kb add "content" -t TYPE -p PROJECT --tags T1,T2 --summary "dense one-liner"
Types: success|failure|experiment|discovery|correction
Tags (confidence): proven|heuristic|open-problem  (importance): core-result|technique|detail
kb-down fallback: ~/.claude/pending-kb-adds/<UTC>.txt with # type/project/tags header; kb flush-pending drains it.
CONVENTIONS

exit 0
