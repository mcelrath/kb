#!/usr/bin/env python3
"""Stop hook (kb-2os.7): surface OUTBOUND pending replies — messages *I* sent with
needs_reply=true that NO peer has answered yet — reading the feed from the
kb-SERVER (GET /bridge/messages).

This is the inverse of bridge-owed-reply-stop.py (which surfaces INBOUND replies I
OWE). Here a "pending reply" = a message where: sender == me, needs_reply is true,
it isn't superseded, and NO message (from anyone) has reply_to == its id. It rebuilds
the `bridge pending-replies` half of the retired bridge-unread-stop.sh on the
kb-server feed.

ADVISORY ONLY (exit 0, additionalContext) — unlike the inbound owed hook there is no
hard-block: a peer not having replied yet is informational, not something I can
resolve by acting. Identity resolves via the registry (agents.json by session_id).
"""
import sys
import os
import json
import urllib.request

BASE = os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765")
AGENTS = os.path.expanduser("~/.agent-bridge/agents.json")


def my_id(stdin_payload) -> str:
    aid = os.environ.get("AGENT_ID", "").strip()
    if aid:
        return aid
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

    # Ids that have been replied to (by anyone) or superseded.
    replied, superseded = set(), set()
    for m in msgs:
        rt = m.get("reply_to")
        if rt not in (None, "None", ""):
            replied.add(str(rt))
        sup = m.get("supersedes")
        if sup not in (None, "None", ""):
            superseded.add(str(sup))

    pending = []
    for m in msgs:
        if m.get("sender") != me:
            continue
        nr = m.get("needs_reply")
        if not (nr is True or str(nr) == "True"):
            continue
        mid = str(m.get("id"))
        if mid in replied or mid in superseded:
            continue
        pending.append(m)
    if not pending:
        return

    def parse_to(s):
        if isinstance(s, list):
            return ",".join(str(x) for x in s)
        return str(s)

    lines = [f"⌛ BRIDGE_PENDING_REPLIES ({len(pending)}) — messages YOU sent with "
             f"--needs-reply that no peer has answered yet (you are owed a reply):"]
    lines += [f"  #{m.get('id')} to {parse_to(m.get('to'))}: {str(m.get('subject'))[:70]}"
              for m in pending]
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "Stop", "additionalContext": "\n".join(lines)}}))
    sys.exit(0)


if __name__ == "__main__":
    main()
