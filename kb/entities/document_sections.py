"""
Document Sections Repository

Manages hierarchical chunks of ingested documents.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

from .base import EntityRepository


VALID_KINDS = ("prose", "table", "figure")


class DocumentSectionsRepository(EntityRepository):
    """Repository for document section management."""

    def __init__(self, conn: sqlite3.Connection, embedding_service: Any | None = None):
        super().__init__(conn)
        # Optional: when set, add() embeds each section into document_sections_vec
        # inline (so ingested docs are searchable immediately, no separate reembed).
        # Left None for schema-only / non-embedding contexts (e.g. unit tests).
        self.embedding_service = embedding_service

    def add(
        self,
        document_id: str,
        path: str,
        level: int,
        ordinal: int,
        kind: str = "prose",
        heading: str | None = None,
        content: str | None = None,
        table_repr: str | None = None,
        embed_text: str | None = None,
        summary: str | None = None,
        token_count: int | None = None,
        content_hash: str | None = None,
        parent_section_id: str | None = None,
        asset_path: str | None = None,
    ) -> str:
        """Add a document section, returning the generated section id."""
        if kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind: {kind!r}. Must be one of {VALID_KINDS}")

        # For documents, the section/subsection TITLE is a better one-line summary
        # than an (absent) LLM summary — and avoids display paths falling back to
        # raw content (e.g. a table's HTML). Default summary to the heading.
        if summary is None and heading:
            summary = heading

        # Embed BEFORE opening the write transaction. The embed is a multi-second
        # network call (with retries); doing it between INSERT and commit would
        # hold the SQLite write lock the whole time, blocking every other kb/kbt
        # writer past busy_timeout ("database is locked") for the entire ingest.
        # Best-effort: a transient embed failure must not abort a large ingest —
        # the row is still stored and can be reembedded later.
        emb = None
        if self.embedding_service is not None:
            text = (embed_text or content or heading or "").strip()
            if text:
                try:
                    emb = self.embedding_service.embed(text)
                except Exception:
                    emb = None  # embed server down/slow — store row, defer vec to reembed

        now = datetime.now().isoformat()
        section_id = f"sec-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"

        # Narrow write transaction: only the two fast inserts + commit hold the lock.
        _ = self.conn.execute(
            """INSERT INTO document_sections
               (id, document_id, parent_section_id, level, ordinal, heading, path,
                content, kind, table_repr, embed_text, summary, token_count,
                content_hash, asset_path, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (section_id, document_id, parent_section_id, level, ordinal, heading, path,
             content, kind, table_repr, embed_text, summary, token_count,
             content_hash, asset_path, now),
        )
        if emb is not None:
            self.conn.execute(
                "DELETE FROM document_sections_vec WHERE id = ?", (section_id,))
            self.conn.execute(
                "INSERT INTO document_sections_vec (id, embedding) VALUES (?, ?)",
                (section_id, emb),
            )

        self.conn.commit()
        return section_id

    def get(self, section_id: str) -> dict[str, object] | None:
        """Get a section by id."""
        row = self.conn.execute(
            "SELECT * FROM document_sections WHERE id = ?", (section_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_by_document(self, document_id: str) -> list[dict[str, object]]:
        """Return all sections for a document, ordered by ordinal."""
        rows = self.conn.execute(
            """SELECT * FROM document_sections
               WHERE document_id = ? AND status = 'active'
               ORDER BY ordinal""",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def breadcrumb(self, section_id: str) -> list[dict[str, object]]:
        """Return ancestor chain root→leaf (inclusive of section_id).

        Each element has keys: id, heading, path, level.
        """
        chain: list[dict[str, object]] = []
        current_id: str | None = section_id
        seen: set[str] = set()

        while current_id is not None:
            if current_id in seen:
                break  # cycle guard
            seen.add(current_id)

            row = self.conn.execute(
                "SELECT id, heading, path, level, parent_section_id FROM document_sections WHERE id = ?",
                (current_id,),
            ).fetchone()
            if not row:
                break

            chain.append({
                "id": row["id"],
                "heading": row["heading"],
                "path": row["path"],
                "level": row["level"],
            })
            current_id = row["parent_section_id"]

        chain.reverse()  # root first
        return chain

    def supersede(self, section_id: str, new_section_id: str) -> bool:
        """Mark a section as superseded by new_section_id."""
        cursor = self.conn.execute(
            """UPDATE document_sections
               SET status = 'superseded', superseded_by = ?
               WHERE id = ?""",
            (new_section_id, section_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def upsert_by_path(
        self,
        document_id: str,
        path: str,
        content_hash: str,
        **kwargs: object,
    ) -> tuple[str, bool]:
        """Insert or supersede a section keyed on (document_id, path).

        Returns (section_id, created) where created=True means a new row was
        inserted (no prior active section at this path, or content changed).
        If an active row exists with the same content_hash, returns it unchanged
        (created=False).
        """
        existing = self.conn.execute(
            """SELECT id, content_hash FROM document_sections
               WHERE document_id = ? AND path = ? AND status = 'active'""",
            (document_id, path),
        ).fetchone()

        if existing is not None:
            if existing["content_hash"] == content_hash:
                # unchanged — skip
                return existing["id"], False
            # content changed — supersede the old row
            old_id = existing["id"]
            new_id = self.add(
                document_id=document_id,
                path=path,
                content_hash=content_hash,
                **kwargs,  # type: ignore[arg-type]
            )
            self.supersede(old_id, new_id)
            return new_id, True

        # no prior active section at this path
        new_id = self.add(
            document_id=document_id,
            path=path,
            content_hash=content_hash,
            **kwargs,  # type: ignore[arg-type]
        )
        return new_id, True
