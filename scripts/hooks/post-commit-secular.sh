#!/bin/sh
# Canonical post-commit hook for ~/Physics/secular-constraints (python KB ingest).
# Symlinked from: ~/Physics/secular-constraints/.git/hooks/post-commit
# Maintained in: ~/Projects/ai/kb/scripts/hooks/post-commit-secular.sh

ROOT=$(git rev-parse --show-toplevel)
KB="$HOME/.local/bin/kb"
LOG="$HOME/Physics/loogle/.rebuild.log"

PY_F=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | grep '\.py$' | sed "s|^|$ROOT/|" | tr '\n' ' ')
[ -n "$PY_F" ] && setsid nohup sh -c "$KB ingest python --files $PY_F" >>"$LOG" 2>&1 </dev/null &
