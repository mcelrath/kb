"""
Errors Repository

Manages error tracking and solution linking.
"""

import os
import sqlite3
from datetime import datetime
from typing import Callable

from .base import EntityRepository


class ErrorsRepository(EntityRepository):
    """Repository for error management."""

    normalize_signature: Callable[[str], str] | None

    def __init__(
        self,
        conn: sqlite3.Connection,
        normalize_signature: Callable[[str], str] | None = None,
    ):
        super().__init__(conn)
        self.normalize_signature = normalize_signature

    def add(
        self,
        signature: str,
        error_type: str | None = None,
        project: str | None = None,
        auto_normalize: bool = True,
    ) -> dict[str, object]:
        """Record an error signature.

        If the error already exists, increments occurrence_count and updates last_seen.

        Returns:
            dict with 'id', 'normalized', 'is_new', 'occurrence_count'
        """
        result: dict[str, object] = {
            "id": None,
            "normalized": False,
            "original_signature": signature,
            "is_new": True,
            "occurrence_count": 1,
        }

        # Auto-normalize the signature
        if auto_normalize and self.normalize_signature:
            normalized = self.normalize_signature(signature)
            if normalized and normalized != signature:
                signature = normalized
                result["normalized"] = True

        now = datetime.now().isoformat()

        # Check if error already exists
        existing = self.conn.execute(
            "SELECT id, occurrence_count FROM errors WHERE signature = ? AND project IS ?",
            (signature, project)
        ).fetchone()

        if existing:
            _ = self.conn.execute(
                "UPDATE errors SET last_seen = ?, occurrence_count = ? WHERE id = ?",
                (now, existing["occurrence_count"] + 1, existing["id"])
            )
            self.conn.commit()
            result["id"] = existing["id"]
            result["is_new"] = False
            result["occurrence_count"] = existing["occurrence_count"] + 1
            return result

        # Create new error
        error_id = f"err-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        _ = self.conn.execute(
            """INSERT INTO errors (id, signature, error_type, project, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (error_id, signature, error_type, project, now, now)
        )
        self.conn.commit()
        result["id"] = error_id
        return result

    def link(
        self,
        error_id: str,
        finding_id: str,
        verified: bool = False,
    ) -> bool:
        """Link an error to a solution (finding).

        Returns True if link was created, False if it already exists.
        """
        now = datetime.now().isoformat()

        try:
            _ = self.conn.execute(
                """INSERT INTO error_solutions (error_id, finding_id, linked_at, verified)
                   VALUES (?, ?, ?, ?)""",
                (error_id, finding_id, now, 1 if verified else 0)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def verify(self, error_id: str, finding_id: str) -> bool:
        """Mark a solution as verified for an error."""
        cursor = self.conn.execute(
            "UPDATE error_solutions SET verified = 1 WHERE error_id = ? AND finding_id = ?",
            (error_id, finding_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get(self, error_id: str) -> dict[str, object] | None:
        """Get an error by ID with its linked solutions."""
        row = self.conn.execute(
            "SELECT * FROM errors WHERE id = ?", (error_id,)
        ).fetchone()

        if not row:
            return None

        # Get linked solutions
        solutions = self.conn.execute(
            """SELECT es.finding_id, es.linked_at, es.verified, f.content, f.type
               FROM error_solutions es
               JOIN findings f ON es.finding_id = f.id
               WHERE es.error_id = ?
               ORDER BY es.verified DESC, es.linked_at DESC""",
            (error_id,)
        ).fetchall()

        return {
            "id": row["id"],
            "signature": row["signature"],
            "error_type": row["error_type"],
            "project": row["project"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "occurrence_count": row["occurrence_count"],
            "solutions": [
                {
                    "finding_id": s["finding_id"],
                    "content": s["content"],
                    "type": s["type"],
                    "verified": bool(s["verified"]),
                    "linked_at": s["linked_at"],
                }
                for s in solutions
            ],
        }

    def search(
        self,
        query: str,
        project: str | None = None,
    ) -> list[dict[str, object]]:
        """Search errors by signature pattern."""
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        sql = "SELECT * FROM errors WHERE signature LIKE ? ESCAPE '\\'"
        params: list[object] = [pattern]

        if project:
            sql += " AND project = ?"
            params.append(project)

        sql += " ORDER BY last_seen DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "occurrence_count": row["occurrence_count"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def list(
        self,
        project: str | None = None,
        error_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """List errors with optional filters."""
        sql = "SELECT * FROM errors WHERE 1=1"
        params: list[object] = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if error_type:
            sql += " AND error_type = ?"
            params.append(error_type)

        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "occurrence_count": row["occurrence_count"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def get_solutions(self, error_id: str) -> list[dict[str, object]]:
        """Get all solutions linked to an error."""
        rows = self.conn.execute(
            """SELECT es.*, f.content, f.type, f.evidence
               FROM error_solutions es
               JOIN findings f ON es.finding_id = f.id
               WHERE es.error_id = ?
               ORDER BY es.verified DESC, es.linked_at DESC""",
            (error_id,)
        ).fetchall()

        return [
            {
                "finding_id": row["finding_id"],
                "content": row["content"],
                "type": row["type"],
                "evidence": row["evidence"],
                "verified": bool(row["verified"]),
                "linked_at": row["linked_at"],
            }
            for row in rows
        ]

    def get_errors_for_solution(self, finding_id: str) -> list[dict[str, object]]:
        """Get all errors that a solution (finding) fixes."""
        rows = self.conn.execute(
            """SELECT e.*, es.verified, es.linked_at
               FROM errors e
               JOIN error_solutions es ON e.id = es.error_id
               WHERE es.finding_id = ?
               ORDER BY es.linked_at DESC""",
            (finding_id,)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "verified": bool(row["verified"]),
                "linked_at": row["linked_at"],
            }
            for row in rows
        ]

    def delete(self, error_id: str) -> bool:
        """Delete an error and its solution links."""
        _ = self.conn.execute("DELETE FROM error_solutions WHERE error_id = ?", (error_id,))
        cursor = self.conn.execute("DELETE FROM errors WHERE id = ?", (error_id,))
        self.conn.commit()
        return cursor.rowcount > 0
