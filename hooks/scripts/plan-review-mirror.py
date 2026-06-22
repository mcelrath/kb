#!/usr/bin/env python3
"""PostToolUse(ExitPlanMode) hook: mirror an APPROVED plan into the project tree
(epic kb-318a8b, kb-fe372c).

Fires only AFTER ExitPlanMode succeeds (i.e. the user actually approved) — so a
cancelled approval never leaves a mirrored artifact (the round-1 stale-mirror fix).
Reads the content-hash verdict marker; if APPROVED, copies the native plan-mode file
(~/.claude/plans/<slug>.md) to <project_root>/.kb/plans/PLAN-<slug>.md so the plan is
committed alongside the code it plans.

No-ops silently unless: tool is ExitPlanMode, kb is on PATH, the marker exists AND is
APPROVED, and project_root is recorded + exists. Never blocks (PostToolUse cannot).
"""

import json
import os
import shutil
import subprocess
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name") != "ExitPlanMode":
        return

    # On PostToolUse, ExitPlanMode puts the path in tool_response.filePath (tool_input
    # is empty here, unlike PreToolUse which carries tool_input.planFilePath).
    ti = data.get("tool_input") or {}
    tr = data.get("tool_response") or {}
    plan_path = ti.get("planFilePath") or tr.get("filePath") or ""
    kb = shutil.which("kb")
    if not kb or not plan_path or not os.path.isfile(plan_path):
        return

    try:
        st = subprocess.run([kb, "plan-review", "status", plan_path],
                            capture_output=True, text=True, timeout=10)
        rec = json.loads((st.stdout or "").strip())
    except Exception:
        return

    if rec.get("verdict") != "APPROVED":
        return
    root = rec.get("project_root", "")
    if not root or not os.path.isdir(root):
        return

    slug = os.path.splitext(os.path.basename(plan_path))[0]
    dest_dir = os.path.join(root, ".kb", "plans")
    dest = os.path.join(dest_dir, f"PLAN-{slug}.md")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(plan_path, dest)
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Approved plan mirrored to {dest}",
        }}))
    except OSError:
        return


if __name__ == "__main__":
    main()
