#!/usr/bin/env python3
"""
Bridge memory precision probe.

Runs 8 realistic queries against the ingested bridge_messages and prints
top hits with scores. Measures whether substantive agent discussion surfaces
from semantic search.

Usage:
    python tmp/bridgemem/probe.py [--db PATH]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_HERE))

import sqlite_vec
from kb.core.embedding import EmbeddingService
from kb.entities.bridge import BridgeMessagesRepository

DEFAULT_DB = Path(_HERE) / "tmp" / "bridgemem" / "bridge.db"

QUERIES = [
    "how to close a kbt issue",
    "opencode async wake test background watcher",
    "embedding dim mismatch reembed force",
    "TS chunker config typescript ingest",
    "GPU wedge persistent kernel barrier UC store",
    "bridge message threading reply protocol",
    "schema migration idempotent init_schema",
    "RDNA3 atomic barrier section 11.4",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = p.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found — run ingest_bridge.py first", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    total, sub, emb = (
        conn.execute("SELECT COUNT(*) FROM bridge_messages").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM bridge_messages WHERE is_substantive=1").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM bridge_messages WHERE embedded=1").fetchone()[0],
    )
    print(f"DB: {total} total, {sub} substantive, {emb} embedded")
    print()

    embedding_service = EmbeddingService()
    repo = BridgeMessagesRepository(conn, embedding_service)

    for query in QUERIES:
        print(f"QUERY: {query!r}")
        results = repo.search(query, limit=3)
        if not results:
            print("  (no results)")
        for r in results:
            body_preview = (r.get("body") or "")[:120].replace("\n", " ").strip()
            print(f"  [{r['id']:4d}] score={r['score']:.3f} sim={r['similarity']:.3f}  "
                  f"{r['sender']} | {r.get('subject', '')[:60]}")
            print(f"         {body_preview}")
        print()


if __name__ == "__main__":
    main()
