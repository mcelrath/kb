#!/bin/bash
# KB Error Extract Hook
# Records error signatures from failed build/test commands to KB.
# Only fires on build/test failures (not general command failures).
# No external LLM — uses simple pattern extraction.

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Resolve venv python
source "${PLUGIN_ROOT}/hooks/scripts/lib/venv-path.sh"
KB_VENV_PYTHON="${KB_VENV_PYTHON:-${KB_VENV_DIR}/bin/python}"
KB_SCRIPT="${PLUGIN_ROOT}/kb.py"

[[ ! -f "$KB_SCRIPT" || ! -x "$KB_VENV_PYTHON" ]] && exit 0

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_name',''))" 2>/dev/null) || exit 0
EXIT_CODE=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('tool_result',{}); print(r.get('exitCode', r.get('exit_code', 0)))" 2>/dev/null) || exit 0

[[ "$TOOL_NAME" != "Bash" ]] && exit 0
[[ "$EXIT_CODE" == "0" ]] && exit 0

# Only fire on build/test commands
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null) || exit 0

IS_BUILD_TEST=0
echo "$COMMAND" | grep -qE '(make|ninja|cmake|cargo build|cargo test|lake build|pytest|python.*test_|python.*-m pytest|python.*-m unittest|g\+\+|gcc|clang|rustc|latexmk|pdflatex)' && IS_BUILD_TEST=1
[[ "$IS_BUILD_TEST" == "0" ]] && exit 0

# Extract error output (last 3000 chars of combined stdout+stderr)
OUTPUT=$(echo "$INPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('tool_result', {})
stdout = r.get('stdout', '') or ''
stderr = r.get('stderr', '') or ''
combined = stdout[-1500:] + stderr[-1500:]
print(combined)
" 2>/dev/null) || exit 0

[[ ${#OUTPUT} -lt 50 ]] && exit 0

# Get project name
if git rev-parse --show-toplevel &>/dev/null; then
    PROJECT=$(basename "$(git rev-parse --show-toplevel)")
else
    PROJECT=$(basename "$PWD")
fi

# Extract first error line (simple pattern match, no LLM)
ERROR_SIG=$(echo "$OUTPUT" | python3 -c "
import sys, re
lines = sys.stdin.readlines()
patterns = [
    r'^.*error\[.*\]:',       # rust
    r'^.*Error:',              # generic
    r'^.*error:',              # gcc/clang/lake
    r'^FAILED',                # pytest
    r'^E\s+',                  # pytest assertion
    r'^.*\.lean:\d+:\d+: error',  # lean
    r'^!.*Error',              # latex
]
for line in lines:
    line = line.strip()
    for p in patterns:
        if re.search(p, line, re.IGNORECASE):
            print(line[:200])
            sys.exit(0)
" 2>/dev/null)

if [[ -n "$ERROR_SIG" ]]; then
    # Record to KB (fire-and-forget, 5s timeout)
    timeout 5 "$KB_VENV_PYTHON" "$KB_SCRIPT" add "BUILD ERROR [$PROJECT]: $ERROR_SIG" \
        -t failure -p "$PROJECT" --tags "build-error" 2>/dev/null && \
        echo "KB: Recorded build error: ${ERROR_SIG:0:80}..."
fi

exit 0
