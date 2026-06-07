#!/bin/sh
# Canonical post-commit hook for ~/Physics/claude (lean + tex KB ingest + loogle rebuild).
# Symlinked from: ~/Physics/claude/.git/hooks/post-commit
# Maintained in: ~/Projects/ai/kb/scripts/hooks/post-commit-claude.sh

HASH=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
ROOT=$(git rev-parse --show-toplevel)
DIR="$ROOT/timestamps"
mkdir -p "$DIR"
echo "$HASH" > "$DIR/$SHORT"
ots stamp "$DIR/$SHORT" 2>/dev/null &

# Loogle rebuild on any .lean commit under proofs/
if git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | grep -q 'proofs/.*\.lean$'; then
  LOOGLE="$HOME/Physics/loogle"
  if [ -x "$LOOGLE/scripts/rebuild-loogle-index.sh" ]; then
    touch "$LOOGLE/.index-dirty"
    setsid nohup sh -c "cd '$LOOGLE' && python scripts/regen_proofs_cache.py >/dev/null 2>&1; '$LOOGLE/scripts/rebuild-loogle-index.sh'" >/dev/null 2>&1 </dev/null &
  fi
fi

# KB incremental ingest
KB="$HOME/.local/bin/kb"
LOG="$HOME/Physics/loogle/.rebuild.log"
CHANGED=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null)

LEAN_F=$(echo "$CHANGED" | grep '\.lean$' | sed "s|^|$ROOT/|" | tr '\n' ' ')
[ -n "$LEAN_F" ] && setsid nohup sh -c "$KB ingest lean --files $LEAN_F" >>"$LOG" 2>&1 </dev/null &

TEX_F=$(echo "$CHANGED" | grep '\.tex$' | sed "s|^|$ROOT/|" | tr '\n' ' ')
[ -n "$TEX_F" ] && setsid nohup sh -c "$KB ingest tex --files $TEX_F" >>"$LOG" 2>&1 </dev/null &
