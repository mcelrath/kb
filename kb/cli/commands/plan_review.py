"""CLI handlers for `kb plan-review` — content-hash-keyed verdict markers that
gate native plan-mode approval on expert-review (epic kb-318a8b).

Transport-agnostic core: any agent host's hook can call these. NO KnowledgeBase /
embedding dependency — dispatched early in kb.py (before DB init) so the
PreToolUse gate stays cheap. Marker store: <kb-cache-dir>/plan-reviews/<hash>.json
where hash = sha256 of the NORMALIZED plan text (see _plan_hash).

Subcommands:
  hash <plan|->            print sha256 of the normalized plan text
  status <plan|->          print the stored verdict JSON for that hash, or 'none'
  prior-rejected <plan|->  exit 0 iff a stored record has this plan's path AND verdict REJECTED
  record <plan|-> ...      write a verdict marker for this plan's hash
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _reviews_dir() -> Path:
    """`<kb-cache-dir>/plan-reviews/`, derived from the configured db_path (no hardcode)."""
    from kb.config import load_config
    d = load_config().db_path.parent / "plan-reviews"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_plan_text(plan_arg: str) -> str:
    """Read plan text from a file path, or stdin when plan_arg == '-'."""
    if plan_arg == "-":
        return sys.stdin.read()
    return Path(plan_arg).read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    """Canonical form for hashing: LF line endings, strip trailing whitespace per
    line, strip trailing newlines. Identical in hash/status/record/prior-rejected —
    any divergence here causes a silent always-re-prompt bug."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).rstrip("\n")


def _plan_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _plan_path(plan_arg: str) -> str:
    """Normalized absolute path of the plan file, or '' for stdin (no identity)."""
    if plan_arg == "-":
        return ""
    return os.path.realpath(plan_arg)


def run_plan_review(args) -> int:
    """Dispatch a `kb plan-review <sub>` call. Returns a process exit code.

    Called EARLY in kb.py (no `kb` KnowledgeBase arg) to avoid embedding-server
    startup — the gate hook invokes status/hash/prior-rejected on every plan exit.
    """
    sub = getattr(args, "plan_review_cmd", None)
    if not sub:
        print("Usage: kb plan-review {hash|status|prior-rejected|record} <plan|->", file=sys.stderr)
        return 2

    try:
        if sub == "hash":
            print(_plan_hash(_read_plan_text(args.plan)))
            return 0

        if sub == "status":
            h = _plan_hash(_read_plan_text(args.plan))
            marker = _reviews_dir() / f"{h}.json"
            if marker.exists():
                print(marker.read_text(encoding="utf-8").strip())
            else:
                print("none")
            return 0

        if sub == "prior-rejected":
            path = _plan_path(args.plan)
            if not path:
                return 1
            for f in _reviews_dir().glob("*.json"):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if rec.get("plan_path") == path and rec.get("verdict") == "REJECTED":
                    return 0
            return 1

        if sub == "record":
            text = _read_plan_text(args.plan)
            h = _plan_hash(text)
            rec = {
                "verdict": args.verdict,
                "synthesis": args.synthesis or "",
                "blocking_issues": list(args.blocking or []),
                "project_root": args.project_root or "",
                "epic_id": args.epic_id or "",
                "plan_path": _plan_path(args.plan),
                "ts": int(time.time()),
            }
            marker = _reviews_dir() / f"{h}.json"
            marker.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(str(marker))
            return 0

    except OSError as e:
        print(f"plan-review {sub}: {e}", file=sys.stderr)
        return 1

    print(f"Unknown plan-review subcommand: {sub}", file=sys.stderr)
    return 2
