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


def _resolve_kb():
    """Resolve the kb CLI WITHOUT depending on the launching env's PATH. A GUI/desktop launch
    or non-login shell often lacks ~/.local/bin, so shutil.which('kb') returns None and the gate
    would silently fail-open (the observed 'inactive' bug). Mirror hooks/scripts/lib/venv-path.sh:
    PATH -> $CLAUDE_PLUGIN_DATA/venv -> ~/.cache/kb/plugin-venv -> $CLAUDE_PLUGIN_ROOT/.venv ->
    ~/.local/bin. First executable wins."""
    cands = [shutil.which("kb")]
    data = os.environ.get("CLAUDE_PLUGIN_DATA")
    if data:
        cands.append(os.path.join(data, "venv", "bin", "kb"))
    cands.append(os.path.expanduser("~/.cache/kb/plugin-venv/bin/kb"))
    root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if root:
        cands.append(os.path.join(root, ".venv", "bin", "kb"))
    cands.append(os.path.expanduser("~/.local/bin/kb"))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


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
    if verdict in ("APPROVED", "APPROVED-WITH-REVISIONS"):
        # Both pass the gate (APPROVED-WITH-REVISIONS = no DESIGN-BLOCKING issues, only
        # implementation notes — dispatch may proceed without a full re-review).
        msg = f"kb:expert-review {verdict} this plan (epic {epic}). {rec.get('synthesis', '')}".strip()
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
    if not rec or rec.get("verdict") not in ("APPROVED", "APPROVED-WITH-REVISIONS"):
        return
    # The execution nudge fires on approval regardless of the mirror copy: bridge plan-approval
    # -> kbt epic -> /dispatch. /decompose-tasks (parent-run) creates the epic+tasks and the parent
    # checks the decomposition before dispatching (hooks can't invoke skills; this is a directive).
    nudge = (f"Plan approved ({rec.get('verdict')}). Next: run `/decompose-tasks {plan_path}` to "
             "create the kbt epic + child tasks, CHECK the decomposition against the plan, then "
             "`/dispatch <epic>`.")
    extra = ""
    root = rec.get("project_root", "")
    if root and os.path.isdir(root):
        slug = os.path.splitext(os.path.basename(plan_path))[0]
        dest = os.path.join(root, ".kb", "plans", f"PLAN-{slug}.md")
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(plan_path, dest)
            extra = f" (plan mirrored to {dest})"
        except OSError:
            pass
    sys.stdout.write(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PostToolUse", "additionalContext": nudge + extra}}))


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name") != "ExitPlanMode":
        return
    event = data.get("hook_event_name")
    kb = _resolve_kb()
    if not kb:
        # VISIBLE fail-open: surface that the gate is OFF rather than silently allowing — this
        # is the failure mode that made the integration look dead. Add a context note (no
        # permissionDecision, so the tool still proceeds — we never hard-block on infra).
        if event == "PreToolUse":
            note = ("plan-review gate INACTIVE: the kb CLI was not found on PATH or in the "
                    "plugin venv, so this plan is NOT being checked against kb:expert-review. "
                    "Fix: `kb configure --install-wrappers` or add ~/.local/bin to PATH.")
            sys.stdout.write(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "additionalContext": note}}))
            sys.stderr.write(note + "\n")
        return
    plan_path = _plan_path(data)
    if not plan_path or not os.path.isfile(plan_path):
        return  # no plan path on the payload -> nothing to gate/mirror
    if event == "PreToolUse":
        _gate(data, kb, plan_path)
    elif event == "PostToolUse":
        _mirror(kb, plan_path)


if __name__ == "__main__":
    main()
