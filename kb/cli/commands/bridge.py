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
    else:
        bridge_parser.print_help()
