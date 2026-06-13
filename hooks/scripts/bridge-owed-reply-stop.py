#!/usr/bin/env python3
"""Stop hook (kb-2os.3): surface INBOUND --needs-reply messages I have NOT replied
to — reading the feed from the kb-SERVER (GET /bridge/messages), not the jsonl.

An "owed reply" = a message where: I am in `to` (broadcasts to 'all' excluded),
needs_reply is true, it isn't superseded, and NO message from me has reply_to ==
its id. Disk/server-derived (recomputed every Stop) so it survives compaction and
re-surfaces until I reply. Default ADVISORY (exit 0); BRIDGE_OWED_HARD_BLOCK=1 or
the owed-hard-block flag file makes it a HARD gate (exit 2). Time-boxed deferral
via $STATE_DIR/owed-deferred (re-blocks after DEFER_TTL).

Migrated from the jsonl-reading version: identical owed/defer/hard-gate logic;
the message source is the kb-server feed and identity resolves via the registry
(agents.json by session_id), not the bridge binary.
"""
import sys
import os
import json
import time
import urllib.request

BASE = os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765")
AGENTS = os.path.expanduser("~/.agent-bridge/agents.json")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from _state import STATE_DIR  # noqa: E402

DEFER_FILE = os.path.join(STATE_DIR, "owed-deferred")
DEFER_TTL = 6 * 3600


def my_id(stdin_payload) -> str:
    aid = os.environ.get("AGENT_ID", "").strip()
    if aid:
        return aid
    # Resolve session_id (from the Stop payload) -> agent id via the registry.
    try:
        sid = json.loads(stdin_payload or "{}").get("session_id", "")
    except Exception:
        sid = ""
    if sid and os.path.exists(AGENTS):
        try:
            reg = json.load(open(AGENTS))
            for a in reg.get("agents", []):
                if a.get("session_id") == sid:
                    return a.get("id", "")
        except Exception:
            pass
    return ""


def parse_to(s) -> list:
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    s = (str(s) if s is not None else "").strip()
    if not s or s == "None":
        return []
    s = s.strip("[]")
    return [p.strip().strip("'\"") for p in s.split(",") if p.strip().strip("'\"")]


def fetch_msgs():
    try:
        with urllib.request.urlopen(f"{BASE}/bridge/messages?limit=500", timeout=4) as r:
            data = json.loads(r.read())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def main():
    try:
        payload = sys.stdin.read()
    except Exception:
        payload = ""
    me = my_id(payload)
    if not me:
        return
    msgs = fetch_msgs()
    if not msgs:
        return

    replied, superseded = set(), set()
    for m in msgs:
        if m.get("sender") == me:
            rt = m.get("reply_to")
            if rt not in (None, "None", ""):
                replied.add(str(rt))
        sup = m.get("supersedes")
        if sup not in (None, "None", ""):
            superseded.add(str(sup))

    owed = []
    for m in msgs:
        nr = m.get("needs_reply")
        if not (nr is True or str(nr) == "True"):
            continue
        if m.get("sender") == me:
            continue
        if me not in parse_to(m.get("to")):
            continue
        mid = str(m.get("id"))
        if mid in replied or mid in superseded:
            continue
        owed.append(m)
    if not owed:
        return

    deferred = {}
    try:
        now = time.time()
        for line in open(DEFER_FILE):
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            ep, did = parts[0], parts[1]
            reason = parts[2].strip() if len(parts) > 2 else ""
            try:
                if now - float(ep) < DEFER_TTL:
                    deferred[did] = reason
            except ValueError:
                pass
    except FileNotFoundError:
        pass

    def fmt(m):
        d = " [deferred]" if str(m.get("id")) in deferred else ""
        return f"  #{m.get('id')} from {m.get('sender')}: {str(m.get('subject'))[:70]}{d}"

    blocking = [m for m in owed if str(m.get("id")) not in deferred]
    HARD_FLAG = os.path.join(STATE_DIR, "owed-hard-block")
    hard = os.environ.get("BRIDGE_OWED_HARD_BLOCK") == "1" or os.path.exists(HARD_FLAG)

    if hard and blocking:
        out = [f"⛔ BRIDGE_OWED_REPLIES — {len(blocking)} unanswered --needs-reply "
               f"message(s) BLOCK idle. For EACH, either:",
               f"  reply:  bridge send <sender> \"<subj>\" --reply <id>  (or POST /bridge/send reply_to=<id>)",
               f"  defer:  echo \"$(date +%s) <id> <why>\" >> {DEFER_FILE}  (re-blocks in {DEFER_TTL//3600}h)"]
        out += [fmt(m) for m in blocking]
        if len(blocking) < len(owed):
            out.append("deferred (still owed, will re-surface):")
            out += [fmt(m) for m in owed if str(m.get("id")) in deferred]
        sys.stderr.write("\n".join(out) + "\n")
        sys.exit(2)

    lines = [f"⚠ BRIDGE_OWED_REPLIES ({len(owed)}) — peer --needs-reply messages you "
             f"have NOT answered. Close with a reply (--reply <id>):"]
    lines += [fmt(m) for m in owed]
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "Stop", "additionalContext": "\n".join(lines)}}))
    sys.exit(0)


if __name__ == "__main__":
    main()
