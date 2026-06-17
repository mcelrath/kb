#!/usr/bin/env python3
"""Stop gate (Phase C, kb-3d347c): block finishing while the final message makes a
claim about a source file this session never READ — inject the unread content/signatures.

Loop safety (REQUIRED — mirrors kbt-inprogress-stop.sh): two layers so a re-wake into a
fresh turn cannot loop forever and exhaust context:
  1. stop_hook_active fast-path.
  2. durable per-session marker — block AT MOST ONCE per session, regardless of
     stop_hook_active (some harnesses reset it each wake).

Detection is bounded: only source files that appear in the LAST assistant message AND are
unread this session AND exist on disk. Best-effort + fail-open: never error out a Stop.
"""
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
try:
    import read_index as ri
    from _state import STATE_DIR, state_path
except Exception:
    sys.exit(0)

_PATH_RX = re.compile(
    r"(?:[\w./~-]+/)?[\w.-]+\.(?:py|rs|ts|tsx|js|jsx|mjs|go|c|h|cpp|cc|hpp|java|rb|lua|sh|lean)"
    r"(?::\d+)?"
)
_DB = os.environ.get("KB_DB") or os.path.expanduser("~/.cache/kb/knowledge.db")


def _last_assistant_text(transcript_path: str) -> str:
    """Concatenate the text of the LAST assistant message in the transcript."""
    last = ""
    try:
        with open(transcript_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "assistant" and rec.get("role") != "assistant":
                    continue
                msg = rec.get("message", rec)
                content = msg.get("content", "")
                if isinstance(content, str):
                    last = content
                elif isinstance(content, list):
                    last = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
    except Exception:
        return ""
    return last


def _signatures(path: str) -> str:
    """Best-effort: symbol signatures+summaries for `path` from the symbols index."""
    try:
        conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT name, signature, docstring_summary FROM symbols WHERE file = ? LIMIT 12",
            (os.path.abspath(path),),
        ).fetchall()
        conn.close()
    except Exception:
        return ""
    if not rows:
        return ""
    out = []
    for name, sig, summ in rows:
        line = f"      {sig or name}"
        if summ:
            line += f"  — {summ[:80]}"
        out.append(line)
    return "\n".join(out)


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0

    if d.get("stop_hook_active") in (True, "true", "True"):
        return 0

    marker = state_path("readidx-stop-blocked")
    if marker and os.path.exists(marker):
        return 0  # already blocked once this session — never loop

    tpath = d.get("transcript_path") or ""
    if not tpath or not os.path.isfile(tpath):
        return 0

    text = _last_assistant_text(tpath)
    if not text:
        return 0

    candidates, seen = [], set()
    for m in _PATH_RX.findall(text):
        p = m.split(":", 1)[0]
        if p in seen:
            continue
        seen.add(p)
        ap = os.path.abspath(os.path.expanduser(p))
        if os.path.isfile(ap) and not ri.is_read(ap):
            candidates.append(ap)

    if not candidates:
        return 0

    # Block ONCE: record the marker first so a re-wake into a fresh turn cannot loop.
    try:
        if marker:
            os.makedirs(STATE_DIR, exist_ok=True)
            open(marker, "w").write("1")
    except Exception:
        pass

    lines = [
        "READ-INDEX BLOCK: your final message makes claims about source files you have NOT",
        "read this session. Read them (Edit/Read) and reconcile your claims against the actual",
        "source before finishing — do not rely on a sub-agent's or memory's description.",
        "",
    ]
    for ap in candidates[:10]:
        lines.append(f"  UNREAD: {ap}")
        sigs = _signatures(ap)
        if sigs:
            lines.append(sigs)
    lines.append("")
    lines.append("(This gate fires once per session; the next Stop is allowed.)")
    sys.stderr.write("\n".join(lines) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
