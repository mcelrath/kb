#!/bin/sh
# Canonical post-commit hook for ~/Physics/mathlib4 (lean KB ingest + loogle rebuild).
# Symlinked from: ~/Physics/mathlib4/.git/hooks/post-commit
# Maintained in: ~/Projects/ai/kb/scripts/hooks/post-commit-mathlib4.sh

# Only act on commits that touched .lean files
git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | grep -q '\.lean$' || exit 0

ROOT=$(git rev-parse --show-toplevel)
LOG="$HOME/Physics/loogle/.rebuild.log"

# Loogle index rebuild (non-blocking, debounced)
REBUILD="$HOME/Physics/loogle/scripts/rebuild-loogle-index.sh"
if [ -x "$REBUILD" ]; then
  touch "$HOME/Physics/loogle/.index-dirty"
  setsid nohup "$REBUILD" >/dev/null 2>&1 </dev/null &
fi

# KB incremental lean ingest
KB="$HOME/.local/bin/kb"
LEAN_F=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | grep '\.lean$' | sed "s|^|$ROOT/|" | tr '\n' ' ')
[ -n "$LEAN_F" ] && setsid nohup sh -c "$KB ingest lean --files $LEAN_F" >>"$LOG" 2>&1 </dev/null &

exit 0
