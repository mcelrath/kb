#!/usr/bin/env python3
"""kb-2os P2: copy the 5 surfacing hooks into the plugin, repointing their lib
imports from ~/.claude/hooks/lib to the plugin-relative lib (portable)."""
import re
import pathlib

SRC = pathlib.Path("/home/mcelrath/Projects/ai/claude/hooks")
DST = pathlib.Path("/home/mcelrath/Projects/ai/kb/hooks/scripts")

pyhooks = [
    "kb/symbol_surface.py",
    "kb/open_issues_surface.py",
    "kb/kb-analysis-surface.py",
    "kb/compose_time_check.py",
]

# (alias).path.expanduser('~/.claude/hooks/lib')  ->  relative-to-this-file/lib
rx = re.compile(r"(\w+)\.path\.expanduser\(\s*['\"]~/\.claude/hooks/lib['\"]\s*\)")


def _repl(m):
    a = m.group(1)
    return f"{a}.path.join({a}.path.dirname({a}.path.abspath(__file__)), 'lib')"


for h in pyhooks:
    text = (SRC / h).read_text()
    new, cnt = rx.subn(_repl, text)
    base = pathlib.Path(h).name
    (DST / base).write_text(new)
    print(f"{base}: repointed {cnt} lib path(s)")

# prior-art-gate.sh (bash): source the plugin-relative ash_health.sh
ph = (SRC / "guards/prior-art-gate.sh").read_text()
ph2 = ph.replace('$HOME/.claude/hooks/lib/ash_health.sh',
                 '$(dirname "$0")/lib/ash_health.sh')
(DST / "prior-art-gate.sh").write_text(ph2)
changed = ph2 != ph
print("prior-art-gate.sh: copied + repointed" if changed
      else "prior-art-gate.sh: copied (NO lib ref changed — check)")
