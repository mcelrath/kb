#!/bin/bash
# KB Pre-Compact Hook
# Extracts findings and context from conversation before /compact
# Uses local LLM (set KB_LLM_URL to override default localhost:9510)

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

LOG_FILE="$HOME/.cache/kb/precompact.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# Get project name
if git rev-parse --show-toplevel &>/dev/null; then
    PROJECT=$(basename "$(git rev-parse --show-toplevel)")
else
    PROJECT=$(basename "$PWD")
fi

log "PreCompact hook started for project: $PROJECT"

# Read conversation from stdin
CONVERSATION=$(cat)
CONV_LEN=${#CONVERSATION}

if [[ $CONV_LEN -lt 1000 ]]; then
    log "Conversation too short ($CONV_LEN chars), skipping"
    exit 0
fi

log "Processing conversation ($CONV_LEN chars)"

# Truncate to last 80k chars for LLM context
if [[ $CONV_LEN -gt 80000 ]]; then
    CONVERSATION="${CONVERSATION: -80000}"
    log "Truncated to 80k chars"
fi

# Set KB environment
export KB_EMBEDDING_URL="${KB_EMBEDDING_URL:-http://ash:8081/embedding}"
export KB_EMBEDDING_DIM=4096
source "${PLUGIN_ROOT}/hooks/scripts/lib/claude-env.sh" 2>/dev/null || true
export KB_LLM_URL="${KB_LLM_URL:-http://${LLM_HOST:-localhost}:${LLM_PORT:-8014}/completion}"

# Resolve venv python
source "${PLUGIN_ROOT}/hooks/scripts/lib/venv-path.sh"
KB_VENV_PYTHON="${KB_VENV_PYTHON:-${KB_VENV_DIR}/bin/python}"
KB_SCRIPT="${PLUGIN_ROOT}/kb.py"

# Gracefully exit if KB tools not installed
if [[ ! -f "$KB_SCRIPT" || ! -x "$KB_VENV_PYTHON" ]]; then
    log "KB tools not installed, skipping"
    exit 0
fi

# Use Python for the LLM call and KB insertion
"$KB_VENV_PYTHON" - "$PROJECT" "$CONVERSATION" << 'PYTHON_SCRIPT'
import sys
import json
import subprocess
import os
from urllib.request import urlopen, Request
from urllib.error import URLError

PROJECT = sys.argv[1]
CONVERSATION = sys.argv[2]

LLM_URL = os.environ.get("KB_LLM_URL", "http://localhost:9510/completion")

# Resolve venv python and kb.py from CLAUDE_PLUGIN_ROOT
_plugin_root = os.environ.get('CLAUDE_PLUGIN_ROOT', '')
if not _plugin_root:
    # Fallback: derive from this script's location (3 levels up from hooks/scripts/)
    _script = os.environ.get('BASH_SOURCE', __file__)
    _plugin_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_script)), '..', '..'))

_data = os.environ.get('CLAUDE_PLUGIN_DATA', '')
if _data:
    KB_VENV_PYTHON = os.path.join(_data, 'venv', 'bin', 'python')
else:
    KB_VENV_PYTHON = os.path.expanduser('~/.cache/kb/plugin-venv/bin/python')

KB_SCRIPT = os.path.join(_plugin_root, 'kb.py')

def llm_complete(prompt: str, max_tokens: int = 2000) -> str | None:
    """Call local LLM for completion using chat API."""
    # Use chat completion endpoint for better format adherence
    chat_url = LLM_URL.replace("/v1/completions", "/v1/chat/completions").replace("/completion", "/v1/chat/completions")
    try:
        req = Request(
            chat_url,
            data=json.dumps({
                "messages": [
                    {"role": "system", "content": "You extract findings from conversations and return ONLY valid JSON. No explanations, no markdown fences, just the JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None

def add_to_kb(content: str, finding_type: str, tags: list[str], evidence: str = "") -> bool:
    """Add a finding to the KB."""
    cmd = [KB_VENV_PYTHON, KB_SCRIPT, "add",
           "-t", finding_type,
           "-p", PROJECT,
           "--force",  # Skip duplicate check - LLM judged significance
           content]

    if tags:
        cmd.extend(["--tags", ",".join(tags)])
    if evidence:
        cmd.extend(["-e", evidence[:500]])  # Truncate evidence

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False

# Extract findings using LLM
EXTRACT_PROMPT = f'''Extract significant technical findings from this conversation.

CONVERSATION:
{CONVERSATION[-60000:]}

Return JSON in exactly this format:
{{"work_context": "one sentence summary", "findings": [{{"type": "success", "content": "what worked", "tags": ["tag1"], "evidence": "quote"}}]}}

Finding types: success (verified working), failure (confirmed broken with reason), discovery (new insight)
Tags: lowercase-hyphenated (gpu-memory, build-error, dim-8)
Maximum 5 findings. Empty array if nothing significant.'''

result = llm_complete(EXTRACT_PROMPT)

if not result:
    print("KB: No LLM response")
    sys.exit(0)

# Debug: log raw LLM response
debug_file = os.path.expanduser('~/.cache/kb/llm_response.txt')
with open(debug_file, "w") as f:
    f.write(result)

# Parse JSON from response
try:
    # Find JSON in response
    json_start = result.find("{")
    json_end = result.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        print("KB: No JSON found in response")
        sys.exit(0)

    json_text = result[json_start:json_end]
    data = json.loads(json_text)
except json.JSONDecodeError as e:
    print(f"KB: JSON parse error: {e}")
    print(f"KB: Raw JSON: {json_text[:500]}")
    sys.exit(0)

# Save work context for post-compact reference
work_context = data.get("work_context", "")
if work_context:
    context_file = os.path.expanduser('~/.cache/kb/last_work_context.txt')
    with open(context_file, "w") as f:
        f.write(f"Project: {PROJECT}\n")
        f.write(f"Context: {work_context}\n")
    print(f"KB: Work context saved: {work_context[:80]}...")

# Process findings
findings = data.get("findings", [])
if not findings:
    print("KB: No significant findings extracted")
    sys.exit(0)

added = 0
for f in findings:
    ftype = f.get("type", "discovery")
    content = f.get("content", "")
    tags = f.get("tags", [])
    evidence = f.get("evidence", "")

    if not content or len(content) < 20:
        continue

    # Validate type
    if ftype not in ("success", "failure", "discovery", "experiment"):
        ftype = "discovery"

    # Ensure tags are strings
    tags = [str(t).lower().replace(" ", "-") for t in tags if t]

    if add_to_kb(content, ftype, tags, evidence):
        added += 1
        print(f"KB: [{ftype.upper()}] {content[:70]}...")

if added > 0:
    print(f"\nKB: Extracted {added} finding(s) before compact")
PYTHON_SCRIPT

EXIT_CODE=$?
log "PreCompact hook completed with exit code $EXIT_CODE"
exit 0  # Always exit 0 to not block compact
