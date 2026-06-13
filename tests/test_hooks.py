"""Regression suite for the kb-plugin hooks (migrated from ~/.claude in kb-2os).

These hooks moved out of the claude-repo harness suite (their tests went with
them). Covered here:
  - block-followup-without-bd-id.sh   : deferral without a tracker-id blocks
  - kbt-lifecycle.sh                  : no-op on non-commit input (id-close needs kbt state)
  - bridge-owed-reply-stop.py         : INBOUND owed replies (mocked kb-server feed)
  - bridge-pending-replies-stop.py    : OUTBOUND pending replies (mocked feed)
  - kb-bridge-watch.sh                : loopback->ash URL rewrite (sandbox reachability)

The two bridge Stop hooks read GET /bridge/messages from the kb-server, so we
stand up a threaded mock server returning canned messages instead of the live one.
"""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks", "scripts")


def _run(argv, stdin="", env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=20, env=e)


# --------------------------------------------------------------------------
# Mock kb-server (serves a canned message list at /bridge/messages)
# --------------------------------------------------------------------------
class _MockServer:
    def __init__(self, messages):
        self.messages = messages
        msgs = self.messages

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/bridge/messages"):
                    body = json.dumps(msgs).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):  # noqa: A002 (match base signature)
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]

    def __enter__(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------
# block-followup-without-bd-id.sh
# --------------------------------------------------------------------------
def _followup_payload(content):
    return json.dumps({"tool_name": "Write",
                       "tool_input": {"file_path": "/x/.claude/plans/PLAN-z.md", "content": content}})


def test_followup_without_id_blocks():
    p = _run(["bash", os.path.join(SCRIPTS, "block-followup-without-bd-id.sh")],
             stdin=_followup_payload("- Strategy B: deferred to a follow-up epic.\n"))
    assert p.returncode == 2, f"expected block (exit 2), got {p.returncode}: {p.stderr}"


def test_followup_with_id_passes():
    p = _run(["bash", os.path.join(SCRIPTS, "block-followup-without-bd-id.sh")],
             stdin=_followup_payload("- kb-1234: sync upstream, deferred to a follow-up epic.\n"))
    assert p.returncode == 0, f"expected pass (exit 0), got {p.returncode}: {p.stderr}"


# --------------------------------------------------------------------------
# kbt-lifecycle.sh  (full bd->kbt close needs live kbt state; assert the no-op path)
# --------------------------------------------------------------------------
def test_kbt_lifecycle_noop_on_non_commit():
    p = _run(["bash", os.path.join(SCRIPTS, "kbt-lifecycle.sh")],
             stdin=json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}}))
    assert p.returncode == 0


# --------------------------------------------------------------------------
# bridge-owed-reply-stop.py  (INBOUND: messages to me, needs_reply, unanswered)
# --------------------------------------------------------------------------
OWED_MSGS = [
    {"id": 1, "sender": "peer", "to": ["me"], "needs_reply": True, "subject": "q1", "body": "b"},
    {"id": 2, "sender": "peer", "to": ["me"], "needs_reply": True, "subject": "q2", "body": "b"},
    {"id": 3, "sender": "me", "to": ["peer"], "needs_reply": False, "subject": "ans", "reply_to": 2, "body": "b"},
    {"id": 4, "sender": "peer", "to": ["other"], "needs_reply": True, "subject": "not-mine", "body": "b"},
]


def test_owed_reply_advisory(tmp_path):
    # Isolate CLAUDE_STATE_DIR so the live ~/.claude/state/owed-hard-block flag
    # (if present) can't turn this advisory case into a hard block.
    with _MockServer(OWED_MSGS) as url:
        p = _run(["python3", os.path.join(SCRIPTS, "bridge-owed-reply-stop.py")],
                 stdin='{"session_id":"x"}',
                 env={"AGENT_ID": "me", "KB_SERVER_URL": url, "CLAUDE_STATE_DIR": str(tmp_path),
                      "BRIDGE_OWED_HARD_BLOCK": ""})  # clear inherited live-session hard-block
    assert p.returncode == 0
    # #1 owed (unanswered to me); #2 answered by #3; #4 to someone else
    assert "#1" in p.stdout and "BRIDGE_OWED_REPLIES" in p.stdout
    assert "#2" not in p.stdout and "not-mine" not in p.stdout


def test_owed_reply_hard_block(tmp_path):
    with _MockServer(OWED_MSGS) as url:
        p = _run(["python3", os.path.join(SCRIPTS, "bridge-owed-reply-stop.py")],
                 stdin='{"session_id":"x"}',
                 env={"AGENT_ID": "me", "KB_SERVER_URL": url,
                      "BRIDGE_OWED_HARD_BLOCK": "1", "CLAUDE_STATE_DIR": str(tmp_path)})
    assert p.returncode == 2, f"hard-block expected exit 2, got {p.returncode}"


def test_owed_reply_defer_clears_block(tmp_path):
    import time
    (tmp_path / "owed-deferred").write_text(f"{int(time.time())} 1 testing\n")
    with _MockServer(OWED_MSGS) as url:
        p = _run(["python3", os.path.join(SCRIPTS, "bridge-owed-reply-stop.py")],
                 stdin='{"session_id":"x"}',
                 env={"AGENT_ID": "me", "KB_SERVER_URL": url,
                      "BRIDGE_OWED_HARD_BLOCK": "1", "CLAUDE_STATE_DIR": str(tmp_path)})
    # only owed id is #1, and it's deferred -> no blocking item -> exit 0
    assert p.returncode == 0, f"defer should clear the block, got {p.returncode}"


# --------------------------------------------------------------------------
# bridge-pending-replies-stop.py  (OUTBOUND: messages I sent, needs_reply, unanswered)
# --------------------------------------------------------------------------
PENDING_MSGS = [
    {"id": 10, "sender": "me", "to": ["peer"], "needs_reply": True, "subject": "my-q", "body": "b"},
    {"id": 11, "sender": "me", "to": ["peer"], "needs_reply": True, "subject": "answered-q", "body": "b"},
    {"id": 12, "sender": "peer", "to": ["me"], "needs_reply": False, "subject": "their-reply", "reply_to": 11, "body": "b"},
    {"id": 13, "sender": "peer", "to": ["me"], "needs_reply": True, "subject": "inbound-not-mine", "body": "b"},
]


def test_pending_replies_outbound():
    with _MockServer(PENDING_MSGS) as url:
        p = _run(["python3", os.path.join(SCRIPTS, "bridge-pending-replies-stop.py")],
                 stdin='{"session_id":"x"}', env={"AGENT_ID": "me", "KB_SERVER_URL": url})
    assert p.returncode == 0
    assert "#10" in p.stdout and "BRIDGE_PENDING_REPLIES" in p.stdout
    # #11 answered by #12; #13 is inbound (not mine to wait on)
    assert "#11" not in p.stdout and "inbound-not-mine" not in p.stdout


def test_pending_replies_none_when_all_answered():
    msgs = [
        {"id": 20, "sender": "me", "to": ["peer"], "needs_reply": True, "subject": "q", "body": "b"},
        {"id": 21, "sender": "peer", "to": ["me"], "needs_reply": False, "subject": "a", "reply_to": 20, "body": "b"},
    ]
    with _MockServer(msgs) as url:
        p = _run(["python3", os.path.join(SCRIPTS, "bridge-pending-replies-stop.py")],
                 stdin='{"session_id":"x"}', env={"AGENT_ID": "me", "KB_SERVER_URL": url})
    assert p.returncode == 0 and p.stdout.strip() == ""


# --------------------------------------------------------------------------
# kb-bridge-watch.sh  (loopback -> ash rewrite; the sandbox-reachability fix)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("given,expect_host", [
    ("http://localhost:8765", "ash"),
    ("http://127.0.0.1:8765", "ash"),
    ("http://ash:8765", "ash"),
    ("http://tardis:9510", "tardis"),
])
def test_watch_loopback_rewrite(given, expect_host):
    # Replicate the script's two rewrite expansions and assert the resolved host.
    snippet = (
        f'BASE="{given}"; '
        'BASE="${BASE//\\/\\/localhost:/\\/\\/ash:}"; '
        'BASE="${BASE//\\/\\/127.0.0.1:/\\/\\/ash:}"; '
        'echo "$BASE"'
    )
    p = _run(["bash", "-c", snippet])
    assert p.returncode == 0
    host = p.stdout.strip().split("//", 1)[1].split(":", 1)[0]
    assert host == expect_host, f"{given} -> {p.stdout.strip()} (host {host}, want {expect_host})"


def test_watch_script_has_rewrite():
    # Drift guard: the parametrized test above replicates the rewrite expansions;
    # assert the real script still carries both, so they can't diverge silently.
    src = open(os.path.join(SCRIPTS, "kb-bridge-watch.sh")).read()
    # The script escapes the slashes in the bash expansion (\/\/localhost:), so
    # match the host tokens that must appear in the rewrite, not literal slashes.
    assert "localhost:" in src and "127.0.0.1:" in src and "ash:" in src


# --------------------------------------------------------------------------
# kb-bridge-watch.sh  (announce frames must NOT wake; everything else does)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("payload,should_wake", [
    ('{"id":1,"event":"announce","subject":"x joined"}', False),
    ('{"id":2,"event":null,"subject":"directed"}', True),
    ('{"id":3,"subject":"no-event-field"}', True),
    ('{"id":4,"event":"hold","subject":"HOLD gpu"}', True),
])
def test_watch_announce_skip(payload, should_wake):
    # Replicate the script's event-extraction + skip decision on one SSE frame.
    snippet = (
        'payload=$1; '
        'event=$(printf "%s" "$payload" | python3 -c '
        '"import sys,json; print((json.load(sys.stdin).get(\'event\') or \'\').strip())" 2>/dev/null); '
        '[ "$event" = "announce" ] && echo SKIP || echo WAKE'
    )
    p = _run(["bash", "-c", snippet, "_", payload])
    got = p.stdout.strip()
    assert got == ("WAKE" if should_wake else "SKIP"), f"{payload} -> {got}"


def test_watch_script_has_announce_skip():
    # Drift guard: the real script must still carry the announce skip.
    src = open(os.path.join(SCRIPTS, "kb-bridge-watch.sh")).read()
    assert 'event' in src and 'announce' in src and 'continue' in src


def test_watch_hold_is_bounded():
    # A run_in_background task has a ~10-min lifetime cap; an unbounded
    # --max-time 86400 hold is KILLED at the cap (exit 144, "failed"). The hold
    # must be bounded under the cap so curl times out cleanly (exit 0). Guards
    # against regressing to the infinite hold.
    src = open(os.path.join(SCRIPTS, "kb-bridge-watch.sh")).read()
    assert "--max-time 86400" not in src, "watcher must not use an infinite hold"
    import re
    m = re.search(r"--max-time\s+(\d+)", src)
    assert m and int(m.group(1)) <= 570, f"watcher --max-time must be < 10-min cap, got {m and m.group(1)}"


def test_inject_kills_watcher_on_userprompt():
    # Drift guard: bridge-inject tears the watcher down on UserPromptSubmit so it
    # is alive only at idle (relaunched at the next Stop).
    src = open(os.path.join(SCRIPTS, "bridge-inject.sh")).read()
    assert 'UserPromptSubmit' in src and 'pkill' in src and 'kb-bridge-watch.sh' in src
