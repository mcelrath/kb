#!/usr/bin/env python3
"""kb-2os.3: fetch unread bridge messages for <id> from the kb-SERVER (not the
jsonl `bridge recv`), advancing a per-session cursor so each message injects once.

Usage:  _bridge_inject_fetch.py <agent-id> <session-id> [event] [is_claude]
  event     = PreToolUse | UserPromptSubmit (omit -> print raw bodies, legacy)
  is_claude = "1" if running under Claude Code (JSON hook-output envelope is
              honored), else "0"/omitted for goose (raw stdout only).

Emits the harness-appropriate hook output (empty if no unread):
  Claude  -> JSON {"systemMessage": "<one user-visible line>",
                   "hookSpecificOutput": {"hookEventName": EVENT,
                                          "additionalContext": "<full bodies>"}}
            so the USER sees a concise "new peer message" banner (systemMessage)
            AND the model still gets the full text (additionalContext). (kb user
            ask: peer messages were model-only/invisible to the human.)
  goose   -> raw bodies on UserPromptSubmit (emit_collect appends them); nothing
            on PreToolUse (emit_blocking has no additionalContext channel).

Tracks the last-injected message id in $STATE_DIR/<session-id>-bridge-injected so
a message injects exactly once (the kb-server GET is stateless — the cursor is ours).
Env: KB_SERVER_URL (default http://127.0.0.1:8765). Any error -> no output.
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765")
STATE_DIR = os.environ.get("KB_STATE_DIR", "/tmp/claude-kb-state")


def _notice(fresh: list) -> str:
    """One concise user-visible line: senders + subjects, capped."""
    parts = []
    for m in fresh:
        sender = m.get("sender", "?")
        subj = (m.get("subject") or "").strip()
        nr = " [needs-reply]" if m.get("needs_reply") else ""
        seg = f"{sender} — {subj}{nr}" if subj else f"{sender}{nr}"
        parts.append(seg)
    joined = "; ".join(parts)
    if len(joined) > 160:
        joined = joined[:157] + "…"
    n = len(fresh)
    prefix = "Peer message received: " if n == 1 else f"Peer messages received ({n}): "
    return prefix + joined


def main():
    if len(sys.argv) < 3:
        return
    agent_id, session_id = sys.argv[1], sys.argv[2]
    event = sys.argv[3] if len(sys.argv) > 3 else ""
    is_claude = (len(sys.argv) > 4 and sys.argv[4] == "1")
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

    # Advance the cursor past EVERY new id (so announces don't re-notify), but only
    # DISPLAY real directed messages — skip ANNOUNCE join frames (event=announce or
    # "ANNOUNCE:" subject), matching the idle watcher's wake filter.
    fresh = []
    max_id = last_id
    for m in msgs:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if mid <= last_id:
            continue
        if mid > max_id:
            max_id = mid
        if (m.get("event") or "").strip() == "announce":
            continue
        if (m.get("subject") or "").strip().upper().startswith("ANNOUNCE:"):
            continue
        fresh.append(m)
    if not fresh:
        # Nothing to show, but still advance the cursor so the filtered announces
        # are not reconsidered next call.
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(cursor_path, "w") as f:
                f.write(str(max_id))
        except Exception:
            pass
        return

    lines = []
    for m in fresh:
        sender = m.get("sender", "?")
        subj = (m.get("subject") or "").strip()
        body = (m.get("body") or "").strip()
        nr = " [needs-reply]" if m.get("needs_reply") else ""
        rt = f" reply-to:#{m['reply_to']}" if m.get("reply_to") else ""
        lines.append(f"[#{m.get('id')}] from={sender}{nr}{rt}  {subj}\n{body}")
    body_text = "\n---\n".join(lines)

    # goose PreToolUse = emit_blocking: no additionalContext channel -> stay silent.
    if event == "PreToolUse" and not is_claude:
        return

    # Advance the cursor only after a successful build (never past undelivered msgs).
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(cursor_path, "w") as f:
            f.write(str(max_id))
    except Exception:
        pass

    wrapped = f"BRIDGE_UPDATE (new peer messages):\n{body_text}\n(end bridge messages)"

    if is_claude and event in ("PreToolUse", "UserPromptSubmit", "SessionStart"):
        notice = _notice(fresh)
        # OSC 9 desktop notification (prefix-free, for terminals that support it:
        # iTerm2/kitty/WezTerm/Ghostty). Strip control chars so they can't break the
        # escape. systemMessage still carries the in-transcript line (renderer prepends
        # "<event> says:" — not suppressible; the OSC popup is the prefix-free path).
        osc_text = "".join(c for c in notice if c not in ("\x1b", "\x07", "\n", "\r"))
        sys.stdout.write(json.dumps({
            "systemMessage": notice,
            "terminalSequence": f"\x1b]9;{osc_text}\x07",
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": wrapped,
            },
        }))
        return

    # goose UserPromptSubmit (emit_collect) / legacy: raw bodies.
    sys.stdout.write(wrapped)


if __name__ == "__main__":
    main()
