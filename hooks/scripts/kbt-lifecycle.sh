#!/bin/bash
# PostToolUse hook for Bash — auto-closes kbt issues referenced in git commit messages
source "$HOME/.claude/hooks/lib/claude-env.sh"

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

[[ "$TOOL_NAME" != "Bash" ]] && exit 0

COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# Only fire on git commit commands
echo "$COMMAND" | grep -qE '^\s*git\s+commit\b' || exit 0

# Check if commit succeeded (PostToolUse payload key is tool_response; fall back to tool_result)
EXIT_CODE=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('tool_response') or d.get('tool_result') or {}; r=r if isinstance(r,dict) else {}; print(r.get('exitCode', r.get('exit_code', 0)))" 2>/dev/null)
[[ "$EXIT_CODE" != "0" ]] && exit 0

# Extract bd issue references from the commit message in the command
BD_IDS=$(echo "$COMMAND" | python3 -c "
import sys, re
cmd = sys.stdin.read()
ids = re.findall(r'(?:fixes|closes|resolves)\s+([a-z]+-[a-z0-9]+)', cmd, re.IGNORECASE)
for i in ids:
    print(i)
" 2>/dev/null)

if [[ -n "$BD_IDS" ]]; then
    while IFS= read -r id; do
        kbt close "$id" 2>/dev/null && echo "Auto-closed kbt issue: $id"
    done <<< "$BD_IDS"
else
    # Warn if implementation files changed but no issue referenced. Use HEAD~1 HEAD
    # (the just-created commit's diff) — NOT --cached HEAD~1, which is ambiguous
    # post-commit (the index equals HEAD, so anything else staged would skew it).
    STAGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -vE '\.(md|txt|json)$' | head -1)
    if [[ -n "$STAGED" ]]; then
        echo "NOTE: Commit touches implementation files but doesn't reference a kbt issue. Consider: fixes <issue-id>"
    fi
fi

exit 0
