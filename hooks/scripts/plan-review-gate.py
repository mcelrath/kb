#!/usr/bin/env python3
"""PreToolUse(ExitPlanMode) gate: blocks native plan-mode approval until the plan
has passed kb:expert-review (epic kb-318a8b).

READ-ONLY by contract — emits only a permission decision, NEVER writes a file (the
project-mirror lives in a separate PostToolUse hook so a cancelled approval leaves
no artifact). Reads the hook stdin JSON, looks up a content-hash-keyed verdict
marker via `kb plan-review`, and emits one of:
  deny  — no marker (never reviewed, or revised since a REJECTED review), or REJECTED
  ask   — APPROVED for this exact plan hash (lets the native dialog show, verdict surfaced)
  (none)— fail-open: kb missing / errored / marker unparseable -> normal approval flow

Decision semantics (Claude Code hooks): permissionDecision deny/ask/allow; we use
deny (block + reason to model) and ask (escalate to the user's approval dialog).
Fail-open emits NOTHING (exit 0) so we never auto-approve a plan on infra failure.
"""

import json
import os
import shutil
import subprocess
import sys


def emit(decision, reason="", context=""):
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    if context:
        out["hookSpecificOutput"]["additionalContext"] = context
    sys.stdout.write(json.dumps(out))
    sys.exit(0)


def _kbpr(kb, *a):
    return subprocess.run([kb, "plan-review", *a], capture_output=True, text=True, timeout=10)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # unparseable input -> defer to normal flow
    if data.get("tool_name") != "ExitPlanMode":
        sys.exit(0)

    plan_path = (data.get("tool_input") or {}).get("planFilePath", "")
    cwd = data.get("cwd", "") or "."

    # project_root for the deny-message dispatch hint (git root of cwd, else cwd).
    root = cwd
    try:
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            root = r.stdout.strip()
    except Exception:
        pass

    kb = shutil.which("kb")
    if not kb or not plan_path or not os.path.isfile(plan_path):
        sys.exit(0)  # fail-open: nothing to gate on

    try:
        st = _kbpr(kb, "status", plan_path)
    except Exception:
        sys.exit(0)  # fail-open: kb errored
    status = (st.stdout or "").strip()

    dispatch = (f"  Task(subagent_type='kb:expert-review', "
                f"plan='{plan_path}', project_root='{root}')")

    if status == "none" or st.returncode != 0:
        try:
            pr = _kbpr(kb, "prior-rejected", plan_path)
            prior_rejected = (pr.returncode == 0)
        except Exception:
            prior_rejected = False
        if prior_rejected:
            emit("deny", reason=(
                "Plan was REVISED since its last review (which was REJECTED). Re-run "
                "kb:expert-review on the new text, address blocking findings, then call "
                "ExitPlanMode again:\n" + dispatch))
        emit("deny", reason=(
            "This plan has not passed kb:expert-review. Run it, address any "
            "DESIGN-BLOCKING findings, then call ExitPlanMode again:\n" + dispatch))

    try:
        rec = json.loads(status)
    except Exception:
        sys.exit(0)  # fail-open: unparseable marker

    verdict = rec.get("verdict")
    epic = rec.get("epic_id", "")
    syn = rec.get("synthesis", "")
    if verdict == "APPROVED":
        msg = f"kb:expert-review APPROVED this plan (epic {epic}). {syn}".strip()
        emit("ask", reason=msg, context=msg)
    if verdict == "REJECTED":
        bl = "; ".join(rec.get("blocking_issues") or [])
        emit("deny", reason=(
            f"kb:expert-review REJECTED this plan (epic {epic}). Blocking: {bl}. "
            "Revise the plan and re-run kb:expert-review before exiting:\n" + dispatch))

    sys.exit(0)  # unknown verdict -> fail-open


if __name__ == "__main__":
    main()
