"""CLI handler for the `bridge` command group.

Uses the bridge-memory layer (bridge_messages + BridgeMessagesRepository,
wired onto the facade as kb._bridge):

    kb bridge ingest [--jsonl PATH] [--since-id N] [--embed-batch N]
    kb bridge search "<query>" [-n N]
    kb bridge promote <id> [-p PROJECT]

ingest:  read ~/.agent-bridge/messages.jsonl, upsert ALL messages into
         bridge_messages, then embed the substantive subset.
search:  hybrid vector+FTS search over substantive messages.
promote: turn a bridge message into a first-class kb FINDING (so durable
         bridge knowledge gets surfaced like any other finding). Uses the
         message subject as the explicit --summary so the LLM path is never hit.
"""

import json
import sys
from pathlib import Path

DEFAULT_JSONL = Path.home() / ".agent-bridge" / "messages.jsonl"


def _load_messages(jsonl_path: Path, since_id: int) -> list[dict]:
    """Parse the bridge jsonl, returning messages with id > since_id."""
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return []
    messages: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: line {lineno} JSON error: {e}", file=sys.stderr)
                continue
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            try:
                msg_id = int(msg_id)
            except (TypeError, ValueError):
                continue
            if msg_id <= since_id:
                continue
            msg["id"] = msg_id
            messages.append(msg)
    return messages


def _run_ingest(kb, args) -> None:
    jsonl = Path(args.jsonl) if getattr(args, "jsonl", None) else DEFAULT_JSONL
    since_id = getattr(args, "since_id", 0) or 0
    embed_batch = getattr(args, "embed_batch", 200) or 200

    messages = _load_messages(jsonl, since_id)
    print(f"Loaded {len(messages)} messages from {jsonl} (since_id={since_id})")

    new_count = 0
    sub_new = 0
    skipped = 0
    for msg in messages:
        result = kb._bridge.upsert(msg)
        if result["is_new"]:
            new_count += 1
            if result["is_substantive"]:
                sub_new += 1
        else:
            skipped += 1
    kb.conn.commit()
    print(f"Upserted: {new_count} new ({sub_new} substantive), {skipped} already present")

    print(f"Embedding up to {embed_batch} substantive messages...")
    embedded = kb._bridge.embed_pending(limit=embed_batch)
    print(f"Embedded: {embedded}")

    counts = kb._bridge.count()
    print(f"DB totals: {counts['total']} messages, {counts['substantive']} substantive, "
          f"{counts['embedded']} embedded")


def _run_search(kb, args) -> None:
    limit = getattr(args, "limit", 10) or 10
    results = kb._bridge.search(args.query, limit=limit)
    if not results:
        print("No matching bridge messages.")
        return
    for r in results:
        snippet = (r.get("body") or "").replace("\n", " ")[:120]
        sim = r.get("similarity", 0)
        subject = r.get("subject") or "(no subject)"
        print(f"[{r['id']}] {r.get('sender', '?')} ({sim:.2f}) | {subject}")
        print(f"    {snippet}")


def _run_promote(kb, args) -> None:
    msg_id = int(args.id)
    row = kb.conn.execute(
        "SELECT id, ts, sender, subject, body FROM bridge_messages WHERE id = ?",
        (msg_id,),
    ).fetchone()
    if not row:
        print(f"ERROR: bridge message {msg_id} not found", file=sys.stderr)
        sys.exit(1)

    subject = row["subject"] or f"bridge message {msg_id}"
    body = (row["body"] or "").strip()
    sender = row["sender"] or "?"
    if not body:
        print(f"ERROR: bridge message {msg_id} has empty body; nothing to promote",
              file=sys.stderr)
        sys.exit(1)

    content = f"[bridge #{msg_id} from {sender}] {subject}\n\n{body}"
    # Explicit summary (the subject) avoids the LLM summary path; duplicate/
    # contradiction checks also hit the LLM, so disable them for promotion.
    result = kb.add(
        content=content,
        finding_type="discovery",
        project=getattr(args, "project", None),
        tags=["bridge-promoted"],
        summary=subject[:300],
        check_duplicate=False,
        check_contradictions=False,
        auto_tag=False,
        auto_classify=False,
    )
    print(f"Promoted bridge #{msg_id} -> {result['id']}")


def run_bridge(kb, args, bridge_parser) -> None:
    cmd = getattr(args, "bridge_cmd", None)
    if cmd == "ingest":
        _run_ingest(kb, args)
    elif cmd == "search":
        _run_search(kb, args)
    elif cmd == "promote":
        _run_promote(kb, args)
    elif cmd == "watch":
        _run_watch(args)
    elif cmd == "send":
        _run_send(args)
    elif cmd == "recv":
        _run_recv(args)
    elif cmd in ("announce", "join"):
        _run_announce(args)
    elif cmd == "clear-owed":
        _run_clear_owed(args)
    else:
        bridge_parser.print_help()


def _parse_to(s) -> list:
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    s = (str(s) if s is not None else "").strip().strip("[]")
    return [p.strip().strip("'\"") for p in s.split(",") if p.strip().strip("'\"")]


def _run_clear_owed(args) -> None:
    """kb bridge clear-owed [<id>] — clear ALL owed --needs-reply messages for this
    agent (stale backlog from ended peer sessions). Records the ids in a permanent
    owed-cleared set the Stop hook honors; does NOT spam senders with stub replies."""
    import json
    import os
    import urllib.request
    me = (getattr(args, "agent_id", None) or "").strip() or _self_id(args)
    if not me:
        print("kb bridge clear-owed: could not infer your id — pass <id> or run /persona.",
              file=sys.stderr)
        sys.exit(1)
    url = f"{_server_url()}/bridge/messages?recipient={me}&limit=500"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            msgs = json.loads(r.read())
    except Exception as e:
        print(f"kb bridge clear-owed: kb-server unreachable ({e})", file=sys.stderr)
        sys.exit(1)
    replied = {str(m["reply_to"]) for m in msgs
               if m.get("sender") == me and m.get("reply_to") not in (None, "None", "")}
    owed = []
    for m in msgs:
        nr = m.get("needs_reply")
        if not (nr is True or str(nr) == "True"):
            continue
        if m.get("sender") == me:
            continue
        if me not in _parse_to(m.get("to")):       # explicit-to-me (not 'all' broadcasts)
            continue
        mid = str(m.get("id"))
        if mid in replied:
            continue
        owed.append(mid)
    if not owed:
        print(f"kb bridge clear-owed: no owed replies for {me}.")
        return
    sd = _state_dir()
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "owed-cleared"), "a") as f:
        for mid in owed:
            f.write(mid + "\n")
    print(f"kb bridge clear-owed: cleared {len(owed)} owed reply(ies) for {me}.")


def _server_url() -> str:
    import os
    return os.environ.get("KB_SERVER_URL", "http://127.0.0.1:8765").rstrip("/")


def _state_dir() -> str:
    import os
    return os.environ.get("CLAUDE_STATE_DIR") or os.path.expanduser("~/.claude/state")


def _self_id(args=None) -> str:
    """Infer THIS agent's bridge id without --from. Resolution order (mirrors
    bridge-resume.sh): explicit --from -> AGENT_ID env -> persona pin
    (<git-root>/.claude/.persona/session-<CLAUDE_SESSION_ID>) -> `whoami`."""
    import os
    import subprocess
    aid = (getattr(args, "from_id", None) if args else None) or os.environ.get("AGENT_ID", "")
    aid = aid.strip()
    if aid:
        return aid
    sid = os.environ.get("CLAUDE_SESSION_ID", "").strip()
    if sid:
        try:
            root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True).stdout.strip() or os.getcwd()
        except Exception:
            root = os.getcwd()
        pin = os.path.join(root, ".claude", ".persona", f"session-{sid}")
        try:
            with open(pin) as f:
                v = f.read().strip()
            if v:
                return v
        except OSError:
            pass
    binp = os.path.expanduser("~/.agent-bridge/bridge")
    if os.path.exists(binp):
        try:
            out = subprocess.run([binp, "whoami"], capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if line.startswith("Effective identity:"):
                    return line.split(":", 1)[1].strip().split()[0]
        except Exception:
            pass
    return ""


def _run_send(args) -> None:
    """kb bridge send <to> <subject> [--body T|stdin] [--reply N] [--needs-reply] [--from ID]
    POSTs to the kb-server /bridge/send — the canonical send path (NOT the binary)."""
    import json
    import urllib.request
    sender = _self_id(args)
    if not sender:
        print("kb bridge send: could not infer your id (no --from, AGENT_ID, persona "
              "pin, or whoami). Run /persona, or pass --from <id>.", file=sys.stderr)
        sys.exit(1)
    body = args.body
    if body is None:
        body = sys.stdin.read() if not sys.stdin.isatty() else ""
    payload = {"from": sender, "to": args.to, "subject": args.subject or "", "body": body}
    if getattr(args, "reply", None):
        payload["reply_to"] = args.reply
    if getattr(args, "needs_reply", False):
        payload["needs_reply"] = True
    req = urllib.request.Request(
        f"{_server_url()}/bridge/send", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
    except Exception as e:
        print(f"kb bridge send: kb-server unreachable ({e})", file=sys.stderr)
        sys.exit(1)
    if resp.get("ok"):
        print(f"sent: id={resp.get('id')} to={args.to}")
    else:
        print(f"kb bridge send failed: {resp.get('error')}", file=sys.stderr)
        sys.exit(1)


def _run_recv(args) -> None:
    """kb bridge recv [<id>] — drain unread for this agent via the kb-server.
    (Normally unnecessary: peer messages auto-inject every turn; this is an
    explicit on-demand read.)"""
    import json
    import os
    import urllib.request
    rid = (getattr(args, "agent_id", None) or "").strip() or _self_id(args)
    if not rid:
        print("kb bridge recv: could not infer your id — pass <id> or run /persona.",
              file=sys.stderr)
        sys.exit(1)
    limit = getattr(args, "limit", 50) or 50
    url = f"{_server_url()}/bridge/messages?recipient={rid}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            msgs = json.loads(r.read())
    except Exception as e:
        print(f"kb bridge recv: kb-server unreachable ({e})", file=sys.stderr)
        sys.exit(1)
    if not msgs:
        print("(no messages)")
        return
    for m in msgs:
        print(f"[#{m.get('id')}] from {m.get('sender')}: {m.get('subject', '')}")
        b = (m.get("body") or "").strip()
        if b:
            print(f"    {b[:400]}")


def _run_announce(args) -> None:
    """kb bridge announce [...] — registry/identity write. There is no kb-server
    registry-write endpoint yet (full migration is deferred), so this presents the
    existing registry binary UNDER `kb bridge` so agents never invoke it directly.
    All flags/heredoc stdin pass straight through via execvp."""
    import os
    binp = os.path.expanduser("~/.agent-bridge/bridge")
    if not os.path.exists(binp):
        print("kb bridge announce: registry backend not found at "
              f"{binp}", file=sys.stderr)
        sys.exit(1)
    rest = list(getattr(args, "rest", []) or [])
    # Agents naturally type `kb bridge join <id>` — a leading bare token (no dash)
    # is the agent's id; map it to the backend's --id flag.
    if rest and not rest[0].startswith("-"):
        rest = ["--id", rest[0]] + rest[1:]
    os.execvp(binp, [binp, "announce"] + rest)


def _run_watch(args) -> None:
    """Exec the SSE bridge watcher (kb-bridge-watch.sh) for this agent.

    Thin wrapper so `kb bridge watch <id>` is the public interface; the watcher
    script stays the (tested) implementation. Resolves the script via
    CLAUDE_PLUGIN_ROOT, else relative to this file's repo root. os.execvp
    replaces this process so stdout/exit pass through unchanged. Launch with
    run_in_background:true and the timeout param OMITTED (unbounded hold)."""
    import os
    root = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    script = os.path.join(root, "hooks", "scripts", "kb-bridge-watch.sh")
    if not os.path.exists(script):
        print(f"kb bridge watch: watcher not found at {script}", file=sys.stderr)
        sys.exit(1)
    os.execvp("bash", ["bash", script, args.agent_id])
