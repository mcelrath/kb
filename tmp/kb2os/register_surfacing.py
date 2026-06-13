#!/usr/bin/env python3
"""kb-2os P2: register the 5 moved surfacing hooks in the plugin hooks.json and
REMOVE their registrations from ~/.claude/settings.json (exactly-once firing).
Backs up settings.json first; validates both JSON files."""
import json
import os
import shutil

PLUGIN = "/home/mcelrath/Projects/ai/kb/hooks/hooks.json"
SETTINGS = os.path.expanduser("~/.claude/settings.json")
PR = "${CLAUDE_PLUGIN_ROOT}/hooks/scripts"

def cmd_py(name, timeout):
    return {"type": "command", "command": f'"{PR}/{name}"', "timeout": timeout}
def cmd_sh(name, timeout):
    return {"type": "command", "command": f'bash "{PR}/{name}"', "timeout": timeout}

# ---- 1. plugin hooks.json: add matcher-scoped blocks for the moved hooks ----
pj = json.load(open(PLUGIN))
H = pj["hooks"]
H.setdefault("PostToolUse", []).append(
    {"matcher": "Read", "hooks": [cmd_py("symbol_surface.py", 10)]})
H.setdefault("PreToolUse", []).append(
    {"matcher": "Task", "hooks": [cmd_sh("prior-art-gate.sh", 10),
                                  cmd_py("compose_time_check.py", 15),
                                  cmd_py("open_issues_surface.py", 10)]})
H["PreToolUse"].append(
    {"matcher": "Bash", "hooks": [cmd_py("compose_time_check.py", 15),
                                  cmd_py("open_issues_surface.py", 10)]})
H.setdefault("Stop", []).append(
    {"hooks": [cmd_py("kb-analysis-surface.py", 12)]})
json.dump(pj, open(PLUGIN, "w"), indent=2)
print("plugin hooks.json: added symbol_surface(Read), prior-art-gate/compose_time_check/open_issues_surface(Task), compose_time_check/open_issues_surface(Bash), kb-analysis-surface(Stop)")

# ---- 2. settings.json: remove those registrations (by command basename) ----
shutil.copy(SETTINGS, SETTINGS + ".kb2os.bak")
MOVED = {"symbol_surface.py", "prior-art-gate.sh", "compose_time_check.py",
         "open_issues_surface.py", "kb-analysis-surface.py"}
def moved(cmd):
    return any(m in cmd for m in MOVED)

sj = json.load(open(SETTINGS))
removed = 0
for ev, matchers in list(sj.get("hooks", {}).items()):
    newm = []
    for block in matchers:
        kept = [h for h in block.get("hooks", []) if not moved(h.get("command", ""))]
        rm = len(block.get("hooks", [])) - len(kept)
        removed += rm
        if kept:
            block["hooks"] = kept
            newm.append(block)
        # drop blocks that became empty
    sj["hooks"][ev] = newm
json.dump(sj, open(SETTINGS, "w"), indent=2)
print(f"settings.json: removed {removed} moved-hook registration(s); backup at {SETTINGS}.kb2os.bak")

# ---- 3. validate ----
json.load(open(PLUGIN)); json.load(open(SETTINGS))
print("both JSON valid")
