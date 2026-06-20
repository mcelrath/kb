#!/usr/bin/env python3
"""Stop hook (kb-lio): match Claude's ANALYSIS against the kb and surface
near-duplicate prior art when Claude is about to REIMPLEMENT something.

Why this exists: kb-prompt-surface.py matches the USER's prompt; symbol_surface
matches files on Read; compose_time_check matches at dispatch. None of them see
what Claude just *concluded/proposed* in its own response. The classic blind-
reimplementation failure is: Claude analyses a task, says "I'll create X", and X
already exists as a finding it never saw. This hook reads the last assistant
turn, and when it carries reimplementation intent, vector-queries the kb on the
*thing being built* and surfaces strong prior-art.

Guarded to fire RARELY (a blocking Stop hook that fires every turn is hostile):
  1. stop_hook_active  -> fire once per stop (harness lets the 2nd through)
  2. reimplementation-intent gate -> only when about to add/create/build/write
  3. SIM_FLOOR 0.62    -> near-duplicate only (prompt-surface uses 0.42)
  4. dedup: skip findings prompt-surface already injected this turn (kbq:<id>);
     mark our own (kba:<id>) so we never re-surface the same finding.

Mode: ADVISORY by default — logs what it *would* surface to
~/.cache/kb/analysis-surface.log (for fire-rate tuning), exit 0. Set
KB_ANALYSIS_SURFACE_BLOCK=1 to make it BLOCK the stop (exit 2) and inject the
prior-art as guarded continuation. Any failure / embed-down / timeout -> exit 0.
"""
import sys
import os
import json
import subprocess
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
try:
    from _seen import filter_unseen  # noqa: E402
    from _state import state_path  # noqa: E402
except Exception:
    def filter_unseen(keys):
        return keys

    def state_path(_name):
        return None

# Resolve kb.py + the venv python from the plugin root (portable across machines);
# fall back to the dev checkout only if CLAUDE_PLUGIN_ROOT isn't set. The plugin
# venv lives under CLAUDE_PLUGIN_DATA (built by setup-venv.sh); prefer it.
_PR = os.environ.get('CLAUDE_PLUGIN_ROOT', os.path.expanduser('~/Projects/ai/kb'))
_PDATA = os.environ.get('CLAUDE_PLUGIN_DATA', '')
KB_SCRIPT = os.environ.get('KB_SCRIPT', os.path.join(_PR, 'kb.py'))
KB_VENV = os.environ.get(
    'KB_VENV',
    os.path.join(_PDATA, 'venv', 'bin', 'python') if _PDATA
    else os.path.join(_PR, '.venv', 'bin', 'python'),
)
SIM_FLOOR = 0.62          # near-duplicate floor; blocking is intrusive, so be strict
MAX_SURFACE = 2
MIN_ASSISTANT_LEN = 200   # need a substantive analysis to match against
BLOCK = os.environ.get('KB_ANALYSIS_SURFACE_BLOCK') == '1'
LOG_PATH = os.path.expanduser('~/.cache/kb/analysis-surface.log')

# Reimplementation intent: the assistant proposing to BUILD something new.
INTENT_RX = re.compile(
    r"(?i)("
    r"\bi['’]?ll\s+(?:add|create|build|implement|write|wire|introduce|scaffold)\b|"
    r"\blet me\s+(?:add|create|build|implement|write|wire|introduce)\b|"
    r"\bgoing to\s+(?:add|create|build|implement|write|wire)\b|"
    r"\bi['’]?ll\s+write\s+a\b|"
    r"\bnew\s+(?:function|module|hook|class|script|endpoint|helper|command|method|repository|table)\b|"
    r"\bimplement(?:ing)?\s+(?:a|an|the)\b|"
    r"\bcreate\s+(?:a|an|the)\s+\w+|"
    r"\badd\s+(?:a|an|the)\s+\w+\s+(?:function|method|module|hook|endpoint|command|class|helper|table)\b|"
    r"\bwrite\s+(?:a|an|the)\s+\w+\s+(?:function|module|script|hook|helper)\b"
    r")"
)


def _truthy(v):
    return v is True or (isinstance(v, str) and v.lower() == 'true') or v == 1


def _tail_lines(path, n=400):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            read_back = min(size, 2 * 1024 * 1024)
            f.seek(size - read_back)
            data = f.read()
        return data.decode('utf-8', errors='replace').splitlines()[-n:]
    except Exception:
        return []


def _last_assistant_text(path):
    for raw in reversed(_tail_lines(path)):
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        msg = ev.get('message') or ev
        if not isinstance(msg, dict):
            continue
        role = msg.get('role') or ev.get('type') or ev.get('role')
        if role != 'assistant':
            continue
        c = msg.get('content', [])
        if isinstance(c, str):
            return c
        parts = [b.get('text', '') for b in c
                 if isinstance(b, dict) and b.get('type') == 'text']
        if parts:
            return '\n'.join(parts)
    return ''


def _already_seen(key):
    """Read-only membership check (does NOT mark) — used to skip findings the
    prompt-surface hook already injected this turn (kbq:<id>)."""
    p = state_path('hook-seen')
    if not p or not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return key in set(f.read().splitlines())
    except Exception:
        return False


def _intent_window(text):
    """Return a focused query around the reimplementation-intent match (the
    'thing being built'), not the whole preamble."""
    m = INTENT_RX.search(text)
    if not m:
        return None
    start = max(0, m.start() - 40)
    end = min(len(text), m.end() + 280)
    return text[start:end].strip()


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if _truthy(data.get('stop_hook_active')):
        return  # fire once per stop
    path = data.get('transcript_path', '')
    if not path or not os.path.isfile(path):
        return
    if not (os.path.isfile(KB_SCRIPT) and os.path.isfile(KB_VENV)):
        return

    text = _last_assistant_text(path)
    if len(text) < MIN_ASSISTANT_LEN:
        return
    query = _intent_window(text)
    if not query:
        return  # no reimplementation intent — nothing to guard

    env = dict(os.environ)
    env.setdefault('KB_EMBEDDING_URL', 'http://ash:8081/embedding')
    env.setdefault('KB_EMBEDDING_DIM', '4096')
    try:
        r = subprocess.run(
            [KB_VENV, KB_SCRIPT, 'search', query[:600], '-n', '8', '--json'],
            capture_output=True, text=True, timeout=8, env=env,
        )
    except Exception:
        return
    if r.returncode != 0 or not r.stdout.strip():
        return
    try:
        results = json.loads(r.stdout)
    except Exception:
        return
    if not isinstance(results, list):
        return

    # Strong hits only, and not ones prompt-surface already showed this turn.
    cands = []
    for rec in results:
        try:
            sim = float(rec.get('similarity') or 0)
        except (TypeError, ValueError):
            sim = 0.0
        if sim < SIM_FLOOR:
            continue
        rid = rec.get('id')
        if not rid or _already_seen(f'kbq:{rid}'):
            continue
        cands.append((sim, rec))
    if not cands:
        return
    cands.sort(key=lambda x: -x[0])

    # Mark our own keys so we never re-surface the same finding.
    fresh = set(filter_unseen([f'kba:{rec["id"]}' for _, rec in cands[:MAX_SURFACE * 2]]))
    lines = []
    for sim, rec in cands:
        if len(lines) >= MAX_SURFACE:
            break
        if f'kba:{rec["id"]}' not in fresh:
            continue
        rid = rec.get('id', '?')
        proj = rec.get('project', '?')
        summ = (rec.get('summary') or rec.get('content') or '')[:90]
        lines.append(f'[KB ~{sim:.2f} {rid} ({proj}): {summ}]')
    if not lines:
        return

    # Emit best-LAST (lines are descending-similarity): U-shaped attention favors the
    # recency slot, and the header holds primacy, so the strongest near-duplicate
    # belongs immediately before the guarded continuation.
    lines.reverse()
    body = (
        "PRIOR ART — your analysis proposes building something the kb may already "
        "have. Before implementing, `kb get <id>` and REUSE if it covers your plan "
        "(per CLAUDE.md: search + reuse before any new function):\n"
        + "\n".join(lines)
    )

    if BLOCK:
        print(body, file=sys.stderr)
        sys.exit(2)
    else:
        # Advisory: log what we WOULD surface (for fire-rate tuning), don't block.
        try:
            with open(LOG_PATH, 'a') as f:
                q = query[:120].replace('\n', ' ')
                f.write(f'WOULD-SURFACE intent={q!r} -> ' + ' '.join(
                    f'{rec.get("id")}~{sim:.2f}' for sim, rec in cands[:MAX_SURFACE]) + '\n')
        except Exception:
            pass
        sys.exit(0)


if __name__ == '__main__':
    main()
