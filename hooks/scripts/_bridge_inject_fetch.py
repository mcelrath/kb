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
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from _state import STATE_DIR  # noqa: E402  (canonical reboot-surviving state root)


def _rel_age(ts_iso: str) -> str:
    """'now' / '5m' / '3h' / '5.6d' from an ISO-8601 ts; '' if unparseable."""
    if not ts_iso:
        return ""
    try:
        t = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        sec = (datetime.now(timezone.utc) - t).total_seconds()
    except Exception:
        return ""
    if sec < 90:
        return "now"
    if sec < 5400:
        return f"{int(sec // 60)}m"
    if sec < 172800:
        return f"{int(sec // 3600)}h"
    return f"{sec / 86400:.1f}d"


def _sender_last_seen(sender: str) -> str:
    """Relative age of the sender's bridge cursor mtime (liveness proxy); '' if unknown."""
    try:
        sec = time.time() - os.path.getmtime(os.path.expanduser(f"~/.agent-bridge/{sender}.cursor"))
    except OSError:
        return ""
    if sec < 120:
        return ""  # fresh — don't clutter
    if sec < 5400:
        return f"{int(sec // 60)}m"
    if sec < 172800:
        return f"{int(sec // 3600)}h"
    return f"{sec / 86400:.1f}d"

BASE = os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765")


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


def _fetch(agent_id, mode, since, limit):
    url = f"{BASE}/bridge/messages?recipient={agent_id}&mode={mode}&limit={limit}"
    if since:
        url += f"&since={since}"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            msgs = json.loads(r.read())
    except Exception:
        return None  # distinguish server-error (None) from empty ([])
    return msgs if isinstance(msgs, list) else []


def _is_announce(m):
    return ((m.get("event") or "").strip() == "announce"
            or (m.get("subject") or "").strip().upper().startswith("ANNOUNCE:"))


def _maxid(msgs, base):
    mx = base
    for m in msgs:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if mid > mx:
            mx = mid
    return mx


def main():
    if len(sys.argv) < 3:
        return
    agent_id, session_id = sys.argv[1], sys.argv[2]
    event = sys.argv[3] if len(sys.argv) > 3 else ""
    is_claude = (len(sys.argv) > 4 and sys.argv[4] == "1")
    if not agent_id or not session_id:
        return

    # Two separate cursors so broadcast volume can NEVER cursor-leap a directed
    # message (the kb-1a0078 bug). Directed is fetched UNBOUNDED since its cursor
    # (every one shown, never evicted); broadcasts are windowed (ambient noise,
    # capped). Migrate from the legacy single cursor so we don't replay history.
    dcur = os.path.join(STATE_DIR, f"{session_id}-bridge-injected-directed")
    bcur = os.path.join(STATE_DIR, f"{session_id}-bridge-injected-broadcast")
    legacy = os.path.join(STATE_DIR, f"{session_id}-bridge-injected")

    def _read(p):
        try:
            return int(open(p).read().strip() or "0")
        except Exception:
            return 0

    def _write(p, v):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(p, "w") as f:
                f.write(str(v))
        except Exception:
            pass

    dlast, blast = _read(dcur), _read(bcur)
    if dlast == 0 and blast == 0:
        leg = _read(legacy)
        dlast = blast = leg

    directed = _fetch(agent_id, "directed", dlast, 200)
    broadcast = _fetch(agent_id, "broadcast", blast, 12)
    if directed is None and broadcast is None:
        return  # server unreachable — change nothing
    directed = directed or []
    broadcast = broadcast or []

    dmax = _maxid(directed, dlast)
    bmax = _maxid(broadcast, blast)

    # Diagnostic trace (KB_BRIDGE_INJECT_DEBUG=1): one line per invocation capturing
    # the exact (event, is_claude, per-class since, returned ids) so a re-inject that
    # is not statically reproducible (archie #5970/kb-333773) can be caught in the act.
    if os.environ.get("KB_BRIDGE_INJECT_DEBUG") == "1":
        try:
            dbg = os.path.join(os.path.expanduser("~/.cache/kb"), "bridge-inject-debug.log")
            os.makedirs(os.path.dirname(dbg), exist_ok=True)
            d_ids = [m.get("id") for m in directed]
            b_ids = [m.get("id") for m in broadcast]
            with open(dbg, "a") as f:
                f.write(f"{agent_id} sess={session_id[:8]} event={event} claude={is_claude} "
                        f"dlast={dlast} blast={blast} directed={d_ids} broadcast={b_ids} "
                        f"-> dmax={dmax} bmax={bmax}\n")
        except Exception:
            pass
    # Directed always shown (never evicted); broadcasts shown but ambient. Announce
    # frames advance the cursor (so they don't re-notify) but are not displayed.
    fresh = [m for m in directed if not _is_announce(m)] \
        + [m for m in broadcast if not _is_announce(m)]
    if not fresh:
        _write(dcur, dmax)
        _write(bcur, bmax)
        _write(legacy, max(dmax, bmax))
        return

    lines = []
    for m in fresh:
        sender = m.get("sender", "?")
        subj = (m.get("subject") or "").strip()
        body = (m.get("body") or "").strip()
        nr = " [needs-reply]" if m.get("needs_reply") else ""
        rt = f" reply-to:#{m['reply_to']}" if m.get("reply_to") else ""
        age = _rel_age(m.get("ts", ""))
        agetag = f" ({age} ago)" if age and age != "now" else ""
        ls = _sender_last_seen(sender)
        # Surface staleness so a multi-day-old message isn't mistaken for current and
        # the sender isn't assumed online (cursor mtime is a proxy; absent => no claim).
        stale = f"  ⟨sender last active {ls} ago — may be offline⟩" if ls else ""
        lines.append(f"[#{m.get('id')}] from={sender}{nr}{rt}{agetag}{stale}  {subj}\n{body}")
    body_text = "\n---\n".join(lines)

    # goose PreToolUse = emit_blocking: no additionalContext channel -> stay silent
    # (return BEFORE advancing cursors so we never consume an undeliverable message).
    if event == "PreToolUse" and not is_claude:
        return

    # Advance cursors only after a successful build (never past undelivered msgs).
    _write(dcur, dmax)
    _write(bcur, bmax)
    _write(legacy, max(dmax, bmax))

    wrapped = f"BRIDGE_UPDATE (new peer messages):\n{body_text}\n(end bridge messages)"

    if is_claude and event in ("PreToolUse", "UserPromptSubmit"):
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
