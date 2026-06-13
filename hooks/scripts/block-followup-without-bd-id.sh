#!/bin/bash
# PreToolUse hook for Write. Blocks plan writes that defer work without a tracker-ID (kbt).
#
# Triggers on Write to ~/.claude/plans/PLAN-*.md.  Scans the new content for
# follow-up / deferred / out-of-scope language. For each such reference,
# requires a kbt-ID (e.g., 'llamacpp-abcd', 'bd-1234', '<project>-<short>')
# on the same line or within the next 3 lines. If any reference lacks a
# nearby kbt-ID, the write is blocked.
#
# Rationale: see CLAUDE.md "Follow-up Discipline (no orphan deferrals)".
# Pattern history: plans repeatedly defer load-bearing work into free-text
# follow-up bullets that nobody ever picks up. kbt-IDs make the work
# first-class so 'kbt ready' surfaces it.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

[ "$TOOL_NAME" != "Write" ] && [ "$TOOL_NAME" != "Edit" ] && exit 0

FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Only check plan files. Match ANY .../plans/PLAN-*.md — covers both the
# ~/.claude/plans symlink path AND the real ~/Projects/ai/claude/plans path
# (plans are written via the REAL path to dodge the config-dir edit guard).
case "$FILE_PATH" in
    */plans/PLAN-*.md) ;;
    *) exit 0 ;;
esac

# Write supplies 'content'; Edit supplies 'new_string'. Scan whichever applies.
CONTENT=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('tool_input', {})
print(data.get('content') or data.get('new_string') or '')
" 2>/dev/null)

# bd-ID regex: <project-slug>-<short> or bd-<short>. Examples:
#   llamacpp-abcd, secular-constraints-adkh, bd-1234, claude-xy12
BD_ID_RX='([a-z][a-z0-9_-]+-[a-z0-9]+|bd-[a-z0-9]+)'

# Scan content. For each line containing a follow-up trigger phrase, check
# whether that line OR any of the next 3 lines contains a bd-ID.
VIOLATIONS=$(echo "$CONTENT" | python3 -c "
import sys, re
content = sys.stdin.read()
lines = content.split('\n')

# Trigger phrases — case-insensitive, word-boundary anchored.
trigger_rx = re.compile(
    r'(?i)\b('
    r'out[- ]of[- ]scope|'
    r'follow[- ]up|'
    r'follow[- ]ups|'
    r'deferred?(\s+to)?|'
    r'future\s+(epic|session|work|fix|sprint)|'
    r'later\s+(epic|session)|'
    r'next\s+(epic|sprint)|'
    r'to[- ]do\s+later'
    r')\b'
)

# Detect real bd-IDs only. Pattern: <project>-<short> where <short> is 3+
# chars with at least one DIGIT. This excludes English compound words like
# 'follow-up', 'out-of-scope'. Real bd IDs are hash-like (digits present).
bd_id_rx = re.compile(r'\b(bd-[a-z0-9]+|[a-z][a-z0-9_]*[a-z0-9](?:-[a-z][a-z0-9_]*[a-z0-9])*-[a-z0-9]*[0-9][a-z0-9]*)\b')

# Skip section headers (### Follow-ups (in bd)) — they're the legitimate marker.
# Only flag bullet/sentence-level references.
violations = []
for i, line in enumerate(lines):
    if not trigger_rx.search(line):
        continue
    # Allow if the line is a section heading with '(in bd)' note
    stripped = line.strip()
    if re.match(r'^#+\s.*\(in bd\)', stripped):
        continue
    # Skip pure header lines without colon or hyphen context
    if re.match(r'^#+\s+(out[- ]of[- ]scope|follow[- ]ups?|deferred?)\s*\$', stripped, re.IGNORECASE):
        # Header alone is allowed; the content lines below must have bd-IDs.
        continue
    # Check this line + next 3 lines for a bd-ID
    window = '\n'.join(lines[i:min(i+4, len(lines))])
    if not bd_id_rx.search(window):
        # Trim very long lines for the error report
        excerpt = line if len(line) <= 120 else line[:117] + '...'
        violations.append(f'  line {i+1}: {excerpt}')

if violations:
    print('\n'.join(violations[:10]))  # cap at 10 violations
" 2>/dev/null)

if [ -n "$VIOLATIONS" ]; then
    cat >&2 <<EOF
BLOCKED: plan '$FILE_PATH' contains follow-up / out-of-scope / deferred references
without a kbt-ID anchor.

Per CLAUDE.md "Follow-up Discipline (no orphan deferrals)": every deferred item
must be a real kbt issue, created BEFORE plan submission, with
'--deps=discovered-from:<this-epic-id>'. Plans refer to follow-ups by kbt-ID.

Violations found:
$VIOLATIONS

How to fix:
  1. For each deferred item, create a bd issue:
       kbt create --title="<title>" --type=task --priority=3 \\
                 --deps=discovered-from:<current-epic-id> \\
                 --description="Discovered during <current-epic-id>. <why>."
  2. Replace the free-text bullet in the plan with the bd-ID:
       BEFORE:  - Strategy B (sync mmvq.cu to upstream): deferred to follow-up epic.
       AFTER:   - llamacpp-XXXX: Sync mmvq.cu to upstream — Strategy B for current epic.
  3. Re-run the Write.

If the deferred item is genuinely irrelevant (not load-bearing): delete the
bullet entirely. There is no third category; if you can't prove
non-load-bearing, it's in scope.

This hook is final. Do NOT rephrase to evade the trigger words — that defeats
the purpose. Create the kbt issues.
EOF
    exit 2
fi

exit 0
