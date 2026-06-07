#!/usr/bin/env python3
"""G2: open-issues-surface — PreToolUse on Agent/Bash(bridge send).

When an agent is dispatched or a bridge message is sent, scan the text for
tokens that match open bd issue titles/descriptions. Surface matching issues
as [OPEN-BD: ...] advisories so plans don't drop through compaction cracks.

Fires PreToolUse/Task and PreToolUse/Bash (bridge send only).
Advisory only (exit 0 always).

Token matching strategy: extract significant tokens (>=5 chars) from the
dispatch text; for each open issue, count how many tokens appear in the
issue title+description. Issues with >= MIN_HITS matches surface.
"""
import sys
import json
import os
import re
import subprocess

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _seen import filter_unseen  # noqa: E402

_MIN_TOKEN_LEN = 5
_MIN_HITS = 2        # tokens from prompt that must appear in issue text
_MAX_SURFACE = 5     # max issues to surface per hook call
def _find_bd() -> str:
    """Find bd executable — may be in nvm or ~/.local/bin."""
    import shutil
    path = shutil.which('bd')
    if path:
        return path
    for candidate in [
        os.path.expanduser('~/.local/bin/bd'),
        os.path.expanduser('~/.nvm/versions/node/v24.0.2/bin/bd'),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return 'bd'

_BD = _find_bd()


def _extract_tokens(text: str) -> set[str]:
    """Extract significant lowercase tokens from text."""
    tokens: set[str] = set()
    for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]{4,})\b', text):
        tokens.add(m.group(1).lower())
    return tokens


def _current_repo_root() -> str | None:
    """Return the git repo root of the current working directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, timeout=3,
            cwd=os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd()),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_open_issues() -> list[dict]:
    """Load open bd issues from the CURRENT repo only. Cross-repo surfacing
    caused false-positive advisories (secular-constraints issues in kb sessions)."""
    issues: list[dict] = []
    root = _current_repo_root()
    if not root:
        return issues
    if not os.path.isdir(os.path.join(root, '.beads')):
        return issues
    try:
        result = subprocess.run(
            f'{_BD} list --status=open --json',
            shell=True, capture_output=True, text=True, timeout=5,
            cwd=root,
        )
        if result.returncode == 0 and result.stdout.strip():
            batch = json.loads(result.stdout)
            for item in batch:
                item['_root'] = root
            issues.extend(batch)
    except Exception:
        pass
    return issues


def _score_issue(issue: dict, prompt_tokens: set[str]) -> int:
    """Count how many prompt tokens appear in the issue title+description."""
    haystack = (
        (issue.get('title') or '') + ' ' + (issue.get('description') or '')
    ).lower()
    return sum(1 for tok in prompt_tokens if tok in haystack)


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    ti = data.get('tool_input', {})

    prompt_text = ''
    if tool_name == 'Task' or tool_name == 'Agent':
        prompt_text = ti.get('prompt', '')
    elif tool_name == 'Bash':
        cmd = ti.get('command', '')
        if 'bridge send' not in cmd:
            sys.exit(0)
        # Extract heredoc body + subject line
        parts = []
        m = re.search(r"<<\s*'?EOF'?\s*\n(.+?)(?:\nEOF\b|\Z)", cmd, re.DOTALL)
        if m:
            parts.append(m.group(1))
        m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
        if m2:
            parts.append(m2.group(1))
        prompt_text = '\n'.join(parts)
    else:
        sys.exit(0)

    if not prompt_text or len(prompt_text) < 20:
        sys.exit(0)

    prompt_tokens = _extract_tokens(prompt_text)
    if len(prompt_tokens) < 3:
        sys.exit(0)

    try:
        issues = _load_open_issues()
    except Exception:
        sys.exit(0)

    if not issues:
        sys.exit(0)

    # Score and rank
    scored: list[tuple[int, dict]] = []
    for issue in issues:
        score = _score_issue(issue, prompt_tokens)
        if score >= _MIN_HITS:
            scored.append((score, issue))

    if not scored:
        sys.exit(0)

    scored.sort(key=lambda x: (-x[0], x[1].get('priority', 99)))
    top = scored[:_MAX_SURFACE]

    # Dedup: key = 'bd:{issue_id}'
    keys = [f'bd:{s[1]["id"]}' for s in top]
    new_keys = set(filter_unseen(keys))

    lines = []
    for score, issue in top:
        key = f'bd:{issue["id"]}'
        if key not in new_keys:
            continue
        iid = issue['id']
        title = (issue.get('title') or '?')[:70]
        pri = issue.get('priority', '?')
        status = issue.get('status', 'open')
        lines.append(f'[OPEN-BD: {iid} (P{pri}) — {title}]')

    if lines:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": "\n".join(lines),
            }
        }))

    sys.exit(0)


if __name__ == '__main__':
    main()
