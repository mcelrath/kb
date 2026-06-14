#!/usr/bin/env python3
"""kb-876 Phase 1: make every move-fragile lib reference absolute.

Only RELATIVE refs break when a hook descends into a subdir:
  shell:  "$(dirname "$0")/lib/X"          -> "$HOME/.claude/hooks/lib/X"
  python: dirname(abspath(__file__))/lib   -> expanduser('~/.claude/hooks/lib')
The $HOME/.claude/hooks/lib and expanduser('~/.claude/hooks/lib') refs are
already absolute (lib/ stays put) and are left untouched.

Asserts the expected literal exists in each file before replacing — a missing
match means the inventory is stale and the run aborts (no silent no-op).
"""
import os
import sys

H = os.path.expanduser('~/.claude/hooks')

SHELL = [
    'bd-lifecycle.sh', 'incompleteness-gate.sh', 'incompleteness-scanner.sh',
    'kb-precompact.sh', 'session-followups.sh', 'session-start-resume.sh',
    'kb-context.sh', 'session-init.sh', 'session-precheck.sh',
    'precompact-save-state.sh', 'kb-search-track.sh', 'dedupe-kb-get.sh',
    'prior-art-gate.sh',
]
OLD = '"$(dirname "$0")/lib/'
NEW = '"$HOME/.claude/hooks/lib/'

PY = 'bridge-owed-reply-stop.py'
PY_OLD = "os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')"
PY_NEW = "os.path.expanduser('~/.claude/hooks/lib')"

fail = []
report = []

for f in SHELL:
    p = os.path.join(H, f)
    t = open(p).read()
    n = t.count(OLD)
    if n == 0:
        fail.append(f"{f}: expected literal {OLD!r} not found")
        continue
    open(p, 'w').write(t.replace(OLD, NEW))
    report.append(f"{f}: {n} ref(s) absolute-ized")

p = os.path.join(H, PY)
t = open(p).read()
if PY_OLD not in t:
    fail.append(f"{PY}: expected {PY_OLD!r} not found")
else:
    open(p, 'w').write(t.replace(PY_OLD, PY_NEW))
    report.append(f"{PY}: __file__/lib -> expanduser")

print("\n".join(report))
if fail:
    sys.stderr.write("\nABORTED — unmatched expectations:\n" + "\n".join(fail) + "\n")
    sys.exit(1)
print(f"\nOK: {len(report)} files updated")
