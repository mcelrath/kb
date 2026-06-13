#!/usr/bin/env python3
"""kb-2os.3: fetch unread bridge messages for <id> from the kb-SERVER (not the
jsonl `bridge recv`), advancing a per-session cursor so each message injects once.

Usage:  _bridge_inject_fetch.py <agent-id> <session-id>
Prints the formatted unread bodies to stdout (empty if none). Tracks the
last-injected message id in $STATE_DIR/<session-id>-bridge-injected so a message
is injected exactly once (the kb-server GET is stateless — the cursor is ours).
Env: KB_SERVER_URL (default http://127.0.0.1:8765). Any error -> no output (the
hook stays silent; the old jsonl path is gone but the watcher still covers idle).
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765")
STATE_DIR = os.environ.get("KB_STATE_DIR", "/tmp/claude-kb-state")


def main():
    if len(sys.argv) < 3:
        return
    agent_id, session_id = sys.argv[1], sys.argv[2]
    if not agent_id or not session_id:
        return

    cursor_path = os.path.join(STATE_DIR, f"{session_id}-bridge-injected")
    try:
        last_id = int(open(cursor_path).read().strip() or "0")
    except Exception:
        last_id = 0

    url = f"{BASE}/bridge/messages?recipient={agent_id}&limit=50"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            msgs = json.loads(r.read())
    except Exception:
        return
    if not isinstance(msgs, list):
        return

    fresh = []
    max_id = last_id
    for m in msgs:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if mid <= last_id:
            continue
        fresh.append(m)
        if mid > max_id:
            max_id = mid
    if not fresh:
        return

    lines = []
    for m in fresh:
        sender = m.get("sender", "?")
        subj = (m.get("subject") or "").strip()
        body = (m.get("body") or "").strip()
        nr = " [needs-reply]" if m.get("needs_reply") else ""
        rt = f" reply-to:#{m['reply_to']}" if m.get("reply_to") else ""
        lines.append(f"[#{m.get('id')}] from={sender}{nr}{rt}  {subj}\n{body}")

    # Advance the cursor only after we've successfully built the injection.
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(cursor_path, "w") as f:
            f.write(str(max_id))
    except Exception:
        pass

    sys.stdout.write("\n---\n".join(lines))


if __name__ == "__main__":
    main()
