#!/usr/bin/env python3
"""
Ingest agent bridge messages from ~/.agent-bridge/messages.jsonl into the KB.

ALL messages are stored in bridge_messages (for thread reconstruction).
Only the SUBSTANTIVE subset is embedded into bridge_messages_vec so semantic
search surfaces signal rather than chatter.

Usage:
    python scripts/ingest_bridge.py [--db PATH] [--jsonl PATH] [--since-id N] [--dry-run]

Idempotent: re-running skips rows already in the DB (upsert on id).
Skips re-embedding messages already marked embedded=1.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import sqlite_vec
from kb.core.schema import init_schema
from kb.core.embedding import EmbeddingService
from kb.entities.bridge import BridgeMessagesRepository, is_substantive

DEFAULT_JSONL = Path.home() / ".agent-bridge" / "messages.jsonl"
DEFAULT_DB = Path.home() / ".cache" / "kb" / "knowledge.db"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest bridge messages into KB")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"Bridge messages jsonl (default: {DEFAULT_JSONL})",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"KB database path (default: {DEFAULT_DB})",
    )
    p.add_argument(
        "--since-id",
        type=int,
        default=0,
        metavar="N",
        help="Only process messages with id > N (incremental mode, default: 0 = all)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report without writing to DB",
    )
    p.add_argument(
        "--embed-batch",
        type=int,
        default=200,
        metavar="N",
        help="Embed up to N pending substantive messages per run (default: 200)",
    )
    return p.parse_args()


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Read embedding_meta to determine dim
    try:
        row = conn.execute("SELECT dim FROM embedding_meta WHERE id = 1").fetchone()
        dim = row[0] if row and row[0] else 4096
    except sqlite3.OperationalError:
        dim = 4096
    init_schema(conn, dim)
    return conn


def load_messages(jsonl_path: Path, since_id: int) -> list[dict]:
    messages = []
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)
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


def main() -> None:
    args = parse_args()

    messages = load_messages(args.jsonl, args.since_id)
    print(f"Loaded {len(messages)} messages from {args.jsonl} (since_id={args.since_id})",
          file=sys.stderr)

    if args.dry_run:
        total = len(messages)
        sub_count = sum(1 for m in messages if is_substantive(m))
        print(f"DRY RUN: {total} messages, {sub_count} substantive ({total - sub_count} noise)")
        # Show first 5 substantive
        shown = 0
        for m in messages:
            if is_substantive(m) and shown < 5:
                body_preview = (m.get("body") or "")[:80].replace("\n", " ")
                print(f"  [{m['id']}] {m.get('sender')} | {m.get('subject')} | {body_preview}")
                shown += 1
        return

    conn = open_db(args.db)
    embedding_service = EmbeddingService()
    repo = BridgeMessagesRepository(conn, embedding_service)

    # Upsert all messages
    new_count = 0
    sub_new = 0
    skipped = 0
    for msg in messages:
        result = repo.upsert(msg)
        if result["is_new"]:
            new_count += 1
            if result["is_substantive"]:
                sub_new += 1
        else:
            skipped += 1

    conn.commit()
    print(f"Upserted: {new_count} new ({sub_new} substantive), {skipped} already present",
          file=sys.stderr)

    # Embed pending substantive messages
    print(f"Embedding up to {args.embed_batch} substantive messages...", file=sys.stderr)
    embedded = repo.embed_pending(limit=args.embed_batch)
    print(f"Embedded: {embedded}", file=sys.stderr)

    counts = repo.count()
    print(f"DB totals: {counts['total']} messages, {counts['substantive']} substantive, "
          f"{counts['embedded']} embedded")


if __name__ == "__main__":
    main()
