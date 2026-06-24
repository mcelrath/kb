"""
Bridge Messages Repository

Stores and searches agent bridge messages (from ~/.agent-bridge/messages.jsonl).
ALL messages are stored for thread reconstruction; only the SUBSTANTIVE subset
is embedded so noise (announce/ack/heartbeat) stays out of semantic search.

Standalone: instantiate BridgeMessagesRepository(conn, embedding_service) directly.
Do NOT wire into KnowledgeBase/facade (fenced — other agent owns facade.py).
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService


# Event types that are NEVER substantive regardless of body content.
_NOISE_EVENT_TYPES = frozenset({
    "announce",
    "ack",
    "recv-noop",
    "watch-noop",
    "heartbeat",
    "presence",
    "ping",
    "pong",
})

# Body patterns that indicate non-substantive content even in message/reply events.
# These are checked after stripping whitespace.
# NOTE: Only apply short-body noise patterns when the body is short (< 200 chars).
# A message starting with "ack on X; [real content]" is substantive; a bare "ack" is not.
_BODY_NOISE_PATTERNS_SHORT = [
    # These only fire when body is short (< 200 chars) — pure-ack / pure-ok / watcher signals
    re.compile(r"^ack\b[\s\.\!]*$", re.IGNORECASE),       # bare "ack", "ack.", "ack!"
    re.compile(r"^acknowledged[\s\.\!]*$", re.IGNORECASE),
    re.compile(r"^received[\s\.\!]*$", re.IGNORECASE),
    re.compile(r"^ok[\s\.\!]*$", re.IGNORECASE),
]

_BODY_NOISE_PATTERNS_ANY = [
    # These fire at any body length — always noise
    re.compile(r"^bridge online\b", re.IGNORECASE),
    re.compile(r"^bridge agents:", re.IGNORECASE),  # bridge agents dump
    re.compile(r"^BRIDGE_WAKE\b"),                  # watcher wake-signal lines
]

# Minimum body length (chars, stripped) to be considered substantive.
_MIN_BODY_LEN = 40


def is_substantive(msg: dict[str, Any]) -> bool:
    """Return True if a bridge message carries durable, searchable content.

    Rules (all must pass):

    EXCLUDE always:
      - event_type in {announce, ack, recv-noop, watch-noop, heartbeat, presence, ping, pong}
      - empty or near-empty bodies (< 40 chars after strip)
      - bodies matching known noise patterns: bare ack/ok/received, bridge-agents dump,
        BRIDGE_WAKE lines, "bridge online" announcements

    INCLUDE:
      - event_type message/reply (or unknown/absent) AND body >= 40 chars AND no noise pattern match

    The threshold and noise list are intentionally conservative — false positives
    (embedding noise) are worse than false negatives (missing a borderline message).
    """
    event_type = (msg.get("event_type") or "").lower().strip()

    # Explicit noise event types
    if event_type in _NOISE_EVENT_TYPES:
        return False

    body = (msg.get("body") or "").strip()

    # Too short
    if len(body) < _MIN_BODY_LEN:
        return False

    # Noise body patterns (any length)
    for pat in _BODY_NOISE_PATTERNS_ANY:
        if pat.match(body):
            return False

    # Noise body patterns (short bodies only — a long body starting with "ack on X; ..."
    # may carry real content in what follows; only exclude pure-ack short messages)
    if len(body) < 200:
        for pat in _BODY_NOISE_PATTERNS_SHORT:
            if pat.match(body):
                return False

    # Subject-echo check: body is ONLY the subject text (exact or close match)
    subject = (msg.get("subject") or "").strip()
    if subject and body.lower() == subject.lower():
        return False

    return True


class BridgeMessagesRepository(EntityRepository):
    """Repository for bridge message memory.

    Supports:
    - upsert(msg): store a parsed jsonl message, mark is_substantive
    - search(query, limit): hybrid vector+FTS search over substantive messages
    - get_thread(msg_id): return full thread via reply_to chain
    - embed_pending(limit): embed unembedded substantive messages, write bridge_messages_vec
    """

    embedding_service: EmbeddingService

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self.embedding_service = embedding_service

    def upsert(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Upsert a bridge message row. Idempotent on msg['id'].

        Returns {"id": int, "is_new": bool, "is_substantive": bool}.
        """
        msg_id: int = int(msg["id"])
        ts = msg.get("ts", "")
        sender = msg.get("sender", "")
        # recipients: may be 'to' (list) or absent
        recipients_raw = msg.get("to") or msg.get("recipients")
        if isinstance(recipients_raw, list):
            recipients = json.dumps(recipients_raw)
        elif isinstance(recipients_raw, str):
            recipients = recipients_raw
        else:
            recipients = None
        subject = msg.get("subject")
        body = msg.get("body")
        event_type = msg.get("event_type") or msg.get("type") or "message"
        reply_to_raw = msg.get("reply_to")
        reply_to = int(reply_to_raw) if reply_to_raw is not None else None
        substantive = 1 if is_substantive(msg) else 0
        now = datetime.utcnow().isoformat()

        existing = self.conn.execute(
            "SELECT id, is_substantive, embedded FROM bridge_messages WHERE id = ?",
            (msg_id,),
        ).fetchone()

        if existing:
            return {
                "id": msg_id,
                "is_new": False,
                "is_substantive": bool(existing[1]),
                "embedded": bool(existing[2]),
            }

        self.conn.execute(
            """INSERT INTO bridge_messages
               (id, ts, sender, recipients, subject, body, event_type,
                reply_to, is_substantive, embedded, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (msg_id, ts, sender, recipients, subject, body, event_type,
             reply_to, substantive, now),
        )
        self.conn.commit()  # self-commit like every sibling repo (was: silent loss if caller forgot)
        return {"id": msg_id, "is_new": True, "is_substantive": bool(substantive), "embedded": False}

    def embed_pending(self, limit: int = 500, commit_every: int = 1) -> int:
        """Embed unembedded substantive messages. Returns count embedded.

        Commits every `commit_every` rows (default 1 = per row) so the write
        lock is held only for each row's 3 tiny writes, NOT for the whole run.
        Critical detail: the slow `embedding_service.embed()` network call runs
        BEFORE the row's first DML, so with per-row commit the embed happens
        OUTSIDE any open transaction — the write lock is held microseconds, not
        the ~0.2s/embed. A single end-of-run commit (or a large batch) instead
        holds one write transaction for MINUTES across all the embed calls,
        starving every other writer on the shared DB (concurrent `kbt create`,
        `kb add`, ingest) past their busy_timeout — WAL lets readers through but
        only one writer at a time. This mirrors update_finding_refresh's
        deliberate per-row microsecond-lock pattern.
        """
        rows = self.conn.execute(
            """SELECT id, subject, body FROM bridge_messages
               WHERE is_substantive = 1 AND embedded = 0
               ORDER BY id
               LIMIT ?""",
            (limit,),
        ).fetchall()

        count = 0
        for row in rows:
            msg_id, subject, body = row
            text = f"{subject or ''}\n{body or ''}".strip()
            if not text:
                continue
            try:
                embedding = self.embedding_service.embed(text)
            except Exception:
                continue
            self.conn.execute(
                "DELETE FROM bridge_messages_vec WHERE id = ?", (msg_id,)
            )
            self.conn.execute(
                "INSERT INTO bridge_messages_vec (id, embedding) VALUES (?, ?)",
                (msg_id, embedding),
            )
            self.conn.execute(
                "UPDATE bridge_messages SET embedded = 1 WHERE id = ?", (msg_id,)
            )
            count += 1
            # Release the write lock frequently so other writers aren't starved.
            if count % commit_every == 0:
                self.conn.commit()

        if count:
            self.conn.commit()
        return count

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Hybrid vector + FTS search over substantive bridge messages.

        Uses the same RRF (Reciprocal Rank Fusion) pattern as HybridSearch.
        Degrades to FTS-only if embedding server is unreachable.
        """
        vector_results: dict[int, dict[str, Any]] = {}
        fts_results: dict[int, dict[str, Any]] = {}

        # Vector search
        try:
            query_embedding = self.embedding_service.embed(
                query,
                max_retries=1,
                timeout=10.0,
            )
            rows = self.conn.execute(
                """SELECT m.id, m.ts, m.sender, m.subject, m.body, m.reply_to,
                          v.distance
                   FROM bridge_messages m
                   JOIN bridge_messages_vec v ON m.id = v.id
                   WHERE v.embedding MATCH ? AND k = ?
                   AND m.is_substantive = 1""",
                (query_embedding, limit * 3),
            ).fetchall()
            for rank, row in enumerate(rows, 1):
                dist = float(row[6])
                vector_results[row[0]] = {
                    "row": row,
                    "rank": rank,
                    "similarity": 1 - (dist ** 2) / 2,
                }
        except Exception:
            pass  # degrade to FTS-only

        # FTS search
        fts_query = query.replace('"', '""')
        try:
            fts_rows = self.conn.execute(
                """SELECT m.id, m.ts, m.sender, m.subject, m.body, m.reply_to,
                          fts.rank
                   FROM bridge_messages m
                   JOIN bridge_messages_fts fts ON m.rowid = fts.rowid
                   WHERE bridge_messages_fts MATCH ?
                   AND m.is_substantive = 1
                   ORDER BY fts.rank
                   LIMIT ?""",
                (fts_query, limit * 3),
            ).fetchall()
            for rank, row in enumerate(fts_rows, 1):
                fts_results[row[0]] = {
                    "row": row,
                    "rank": rank,
                    "relevance": -float(row[6]),
                }
        except sqlite3.OperationalError:
            pass

        # RRF merge
        k = 60
        all_ids = set(vector_results.keys()) | set(fts_results.keys())
        merged: list[dict[str, Any]] = []

        for mid in all_ids:
            rrf = 0.0
            row = None
            vec_sim = 0.0
            if mid in vector_results:
                rrf += 1 / (k + vector_results[mid]["rank"])
                row = vector_results[mid]["row"]
                vec_sim = vector_results[mid]["similarity"]
            if mid in fts_results:
                rrf += 1 / (k + fts_results[mid]["rank"])
                if row is None:
                    row = fts_results[mid]["row"]

            if row is None:
                continue

            merged.append({
                "id": row[0],
                "ts": row[1],
                "sender": row[2],
                "subject": row[3],
                "body": (row[4] or "")[:500],
                "reply_to": row[5],
                "similarity": round(vec_sim, 4),
                "score": rrf,
            })

        merged.sort(key=lambda x: x["score"], reverse=True)

        # Normalize scores
        if merged:
            top = merged[0]["score"]
            if top > 0:
                for r in merged:
                    r["score"] = round(r["score"] / top, 4)

        return merged[:limit]

    def get_thread(self, msg_id: int, max_depth: int = 20) -> list[dict[str, Any]]:
        """Return the thread containing msg_id by following reply_to links.

        Walks up to root (reply_to IS NULL) then returns the chain in chronological order.
        Also collects direct replies to the root for a shallow subtree view.
        """
        # Walk up to root
        chain_ids: list[int] = []
        current = msg_id
        seen: set[int] = set()
        for _ in range(max_depth):
            if current in seen:
                break
            seen.add(current)
            chain_ids.append(current)
            row = self.conn.execute(
                "SELECT reply_to FROM bridge_messages WHERE id = ?", (current,)
            ).fetchone()
            if not row or row[0] is None:
                break
            current = row[0]

        # Root is the last in chain_ids; fetch all messages in the thread rooted there
        root_id = chain_ids[-1] if chain_ids else msg_id
        thread_rows = self.conn.execute(
            """SELECT id, ts, sender, subject, body, reply_to, is_substantive
               FROM bridge_messages
               WHERE id = ? OR reply_to = ?
               ORDER BY id""",
            (root_id, root_id),
        ).fetchall()

        results = []
        for r in thread_rows:
            results.append({
                "id": r[0],
                "ts": r[1],
                "sender": r[2],
                "subject": r[3],
                "body": (r[4] or "")[:300],
                "reply_to": r[5],
                "is_substantive": bool(r[6]),
            })
        return results

    def count(self) -> dict[str, int]:
        """Return total and substantive message counts."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM bridge_messages"
        ).fetchone()[0]
        substantive = self.conn.execute(
            "SELECT COUNT(*) FROM bridge_messages WHERE is_substantive = 1"
        ).fetchone()[0]
        embedded = self.conn.execute(
            "SELECT COUNT(*) FROM bridge_messages WHERE embedded = 1"
        ).fetchone()[0]
        return {"total": total, "substantive": substantive, "embedded": embedded}
