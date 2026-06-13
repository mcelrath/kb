#!/bin/bash
# SessionStart hook: surface open follow-ups discovered from recently-closed epics.
#
# Plans frequently defer work as "Out of scope" or "Follow-up" bullets. The
# CLAUDE.md "Follow-up Discipline" rule requires those to be real bd issues
# with --deps=discovered-from:<epic-id>. This hook surfaces those issues at
# session start so they don't fall off the radar between sessions.
#
# Strategy: find all closed epics from the last 30 days, then list any open
# issues that have a 'discovered-from' relationship to those epics.

source "$HOME/.claude/hooks/lib/claude-env.sh" 2>/dev/null

# Bail quietly if bd isn't available or no .beads in cwd or ancestors
command -v kbt >/dev/null 2>&1 || exit 0
kbt list --limit=1 >/dev/null 2>&1 || exit 0

# Get recently closed epics (last 30 days)
RECENT_CLOSED_EPICS=$(kbt list --type=epic --status=closed --json 2>/dev/null | python3 -c "
import sys, json
from datetime import datetime, timedelta, timezone
try:
    items = json.load(sys.stdin)
except Exception:
    sys.exit(0)
cutoff = datetime.now(timezone.utc) - timedelta(days=30)
ids = []
for e in items:
    ts = e.get('updated_at') or e.get('created_at') or ''
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt >= cutoff:
            ids.append(e['id'])
    except Exception:
        continue
print(' '.join(ids))
" 2>/dev/null)

[ -z "$RECENT_CLOSED_EPICS" ] && exit 0

# For each closed epic, find any open issues that depend on it via discovered-from.
FOLLOWUPS=$(
    for epic_id in $RECENT_CLOSED_EPICS; do
        # kbt dep list <epic> shows what depends-on/blocks. We want the inverse:
        # issues that point TO this epic via 'discovered-from'. Use --json on a
        # broad list and filter.
        kbt dep list "$epic_id" --json 2>/dev/null | python3 -c "
import sys, json
try:
    deps = json.load(sys.stdin)
except Exception:
    sys.exit(0)
epic_id = '$epic_id'
# Format varies; iterate and report any dependent/relates with discovered-from type.
items = deps if isinstance(deps, list) else deps.get('dependents', []) + deps.get('depends_on', [])
for d in items:
    if not isinstance(d, dict):
        continue
    dep_type = d.get('type') or d.get('dependency_type') or ''
    if 'discovered' not in dep_type.lower():
        continue
    status = d.get('status', '')
    if status in ('closed', 'cancelled'):
        continue
    iid = d.get('id') or d.get('dependent_id') or d.get('issue_id') or ''
    title = d.get('title', '')
    if iid:
        print(f'  - {iid} (from closed {epic_id}): {title}')
" 2>/dev/null
    done | sort -u | head -10
)

if [ -n "$FOLLOWUPS" ]; then
    echo "FOLLOW-UPS from recently closed epics (open, may have fallen off the radar):"
    echo "$FOLLOWUPS"
    echo "  Run: kbt ready  /  kbt show <id>  to act on these"
fi

exit 0
