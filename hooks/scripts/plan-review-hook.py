#!/usr/bin/env python3
"""ExitPlanMode plan-review hook (epic kb-318a8b). Registered for BOTH PreToolUse
and PostToolUse on ExitPlanMode; branches on hook_event_name.

PreToolUse  = the GATE: block approval until kb:expert-review has recorded an
              APPROVED verdict for the exact plan text. Emits deny/ask only —
              never writes a file, so a cancelled approval leaves no artifact.
PostToolUse = the MIRROR: after a real approval, copy the plan to
              <project_root>/.kb/plans/PLAN-<slug>.md (committed with the code).

Fail-open everywhere (kb missing / errored / marker unparseable -> do nothing,
so we never hard-block or auto-approve a plan on infra failure). Normalization
lives only in `kb plan-review`; the hook shells out to it (never re-hashes).
"""

import json
import os
import shutil
import subprocess
import sys


def _plan_path(data):
    # PreToolUse carries tool_input.planFilePath; PostToolUse carries tool_response.filePath.
    ti = data.get("tool_input") or {}
    tr = data.get("tool_response") or {}
    return ti.get("planFilePath") or tr.get("filePath") or ""


def _status(kb, plan_path):
    """Parsed verdict marker for this plan's hash, or None ('none'/unparseable/error)."""
    try:
        out = subprocess.run([kb, "plan-review", "status", plan_path],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return json.loads(out)
    except Exception:
        return None


def _gate(data, kb, plan_path):
    cwd = data.get("cwd") or "."
    root = cwd
    try:
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            root = r.stdout.strip()
    except Exception:
        pass
    dispatch = f"  Task(subagent_type='kb:expert-review', plan='{plan_path}', project_root='{root}')"

    rec = _status(kb, plan_path)
    if not rec:
        return _emit("deny", "This plan has not passed kb:expert-review. Run it, address any "
                     "DESIGN-BLOCKING findings, then call ExitPlanMode again:\n" + dispatch)
    epic = rec.get("epic_id", "")
    verdict = rec.get("verdict")
    if verdict == "APPROVED":
        msg = f"kb:expert-review APPROVED this plan (epic {epic}). {rec.get('synthesis', '')}".strip()
        return _emit("ask", msg, msg)
    if verdict == "REJECTED":
        bl = "; ".join(rec.get("blocking_issues") or [])
        return _emit("deny", f"kb:expert-review REJECTED this plan (epic {epic}). Blocking: {bl}. "
                     "Revise the plan and re-run kb:expert-review before exiting:\n" + dispatch)
    # unknown verdict -> fail-open (no output)


def _emit(decision, reason="", context=""):
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    if context:
        out["hookSpecificOutput"]["additionalContext"] = context
    sys.stdout.write(json.dumps(out))


def _mirror(kb, plan_path):
    rec = _status(kb, plan_path)
    if not rec or rec.get("verdict") != "APPROVED":
        return
    root = rec.get("project_root", "")
    if not root or not os.path.isdir(root):
        return
    slug = os.path.splitext(os.path.basename(plan_path))[0]
    dest = os.path.join(root, ".kb", "plans", f"PLAN-{slug}.md")
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(plan_path, dest)
        sys.stdout.write(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse", "additionalContext": f"Approved plan mirrored to {dest}"}}))
    except OSError:
        pass


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name") != "ExitPlanMode":
        return
    plan_path = _plan_path(data)
    kb = shutil.which("kb")
    if not kb or not plan_path or not os.path.isfile(plan_path):
        return  # fail-open: nothing to gate/mirror on
    event = data.get("hook_event_name")
    if event == "PreToolUse":
        _gate(data, kb, plan_path)
    elif event == "PostToolUse":
        _mirror(kb, plan_path)


if __name__ == "__main__":
    main()
