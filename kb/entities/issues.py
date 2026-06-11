"""
Issues Repository

Manages issues (tasks, bugs, features, epics, etc.) for the kb-native tracker.
Modeled on ConceptRepository.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService


ISSUE_TYPES = ("task", "bug", "feature", "epic", "chore", "spike", "decision")
ISSUE_STATUSES = ("open", "in_progress", "blocked", "closed")


class IssuesRepository(EntityRepository):
    """Repository for issue tracking (kb-native bd replacement)."""

    embedding_service: EmbeddingService

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self.embedding_service = embedding_service

    def _alloc_child_id(self, parent_id: str) -> str:
        """Allocate next child id for parent_id, atomically via child_counters.

        Uses a single transaction: INSERT OR REPLACE to upsert the counter,
        then reads back the new value. Wrapped in BEGIN IMMEDIATE to prevent
        races under concurrent writers.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT counter FROM child_counters WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
            next_n = (row[0] if row else 0) + 1
            self.conn.execute(
                """INSERT INTO child_counters (parent_id, counter) VALUES (?, ?)
                   ON CONFLICT(parent_id) DO UPDATE SET counter = excluded.counter""",
                (parent_id, next_n),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return f"{parent_id}.{next_n}"

    def create(
        self,
        title: str,
        type: str = "task",
        description: str | None = None,
        status: str = "open",
        priority: int = 2,
        parent_id: str | None = None,
        design_file: str | None = None,
        assignee: str | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
        prefix: str = "kb",
    ) -> dict[str, Any]:
        """Create a new issue.

        Root issues get id `{prefix}-{uuid4().hex[:6]}`.
        Child issues (parent_id given) get `{parent_id}.{N}` where N is
        allocated atomically from child_counters.

        Returns dict with 'id', 'is_new'.
        """
        if type not in ISSUE_TYPES:
            raise ValueError(f"type must be one of {ISSUE_TYPES}")
        if status not in ISSUE_STATUSES:
            raise ValueError(f"status must be one of {ISSUE_STATUSES}")

        if parent_id:
            issue_id = self._alloc_child_id(parent_id)
        else:
            issue_id = f"{prefix}-{uuid.uuid4().hex[:6]}"

        now = datetime.utcnow().isoformat()
        tags_json = json.dumps(tags or [])

        self.conn.execute(
            """INSERT INTO issues
               (id, type, status, priority, parent_id, title, description,
                design_file, assignee, project, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (issue_id, type, status, priority, parent_id, title, description,
             design_file, assignee, project, tags_json, now, now),
        )

        embed_text = title + (" " + description if description else "")
        embedding = self.embedding_service.embed(embed_text)
        self.conn.execute("DELETE FROM issues_vec WHERE id = ?", (issue_id,))
        self.conn.execute(
            "INSERT INTO issues_vec (id, embedding) VALUES (?, ?)",
            (issue_id, embedding),
        )
        self.conn.commit()
        return {"id": issue_id, "is_new": True}

    def get(self, issue_id: str) -> dict[str, Any] | None:
        """Get an issue with its comments and deps."""
        row = self.conn.execute(
            """SELECT id, type, status, priority, parent_id, title, description,
                      design_file, assignee, close_reason, project, tags,
                      created_at, updated_at, started_at, closed_at, closed_by_session
               FROM issues WHERE id = ?""",
            (issue_id,),
        ).fetchone()
        if not row:
            return None

        result = dict(zip([
            "id", "type", "status", "priority", "parent_id", "title", "description",
            "design_file", "assignee", "close_reason", "project", "tags",
            "created_at", "updated_at", "started_at", "closed_at", "closed_by_session",
        ], row))
        result["tags"] = json.loads(result["tags"] or "[]")

        comment_rows = self.conn.execute(
            "SELECT id, body, author, created_at FROM issue_comments WHERE issue_id = ? ORDER BY created_at",
            (issue_id,),
        ).fetchall()
        result["comments"] = [
            dict(zip(["id", "body", "author", "created_at"], r))
            for r in comment_rows
        ]

        dep_rows = self.conn.execute(
            """SELECT issue_id, depends_on_id, type, created_at, created_by
               FROM issue_deps WHERE issue_id = ? OR depends_on_id = ?""",
            (issue_id, issue_id),
        ).fetchall()
        result["deps"] = [
            dict(zip(["issue_id", "depends_on_id", "type", "created_at", "created_by"], r))
            for r in dep_rows
        ]
        return result

    def list(
        self,
        project: str | None = None,
        status: str | None = None,
        type: str | None = None,
        parent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List issues with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if type:
            conditions.append("type = ?")
            params.append(type)
        if parent_id:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = self.conn.execute(
            f"""SELECT id, type, status, priority, parent_id, title, project, tags, created_at
                FROM issues {where} ORDER BY priority, created_at""",
            params,
        ).fetchall()
        results = []
        for r in rows:
            d = dict(zip(["id", "type", "status", "priority", "parent_id", "title", "project", "tags", "created_at"], r))
            d["tags"] = json.loads(d["tags"] or "[]")
            results.append(d)
        return results

    def search(self, query: str, project: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Semantic search over issues via issues_vec (sqlite-vec KNN)."""
        embedding = self.embedding_service.embed(query)
        conditions: list[str] = []
        params: list[Any] = []
        if project:
            conditions.append("i.project = ?")
            params.append(project)

        rows = self.conn.execute(
            f"""SELECT v.id, v.distance
                FROM issues_vec v
                JOIN issues i ON i.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                {"AND " + " AND ".join(conditions) if conditions else ""}
                ORDER BY v.distance""",
            (embedding, limit, *params),
        ).fetchall()

        results = []
        for iid, dist in rows:
            row = self.conn.execute(
                """SELECT id, type, status, priority, parent_id, title, description, project, tags
                   FROM issues WHERE id = ?""",
                (iid,),
            ).fetchone()
            if row:
                r = dict(zip(["id", "type", "status", "priority", "parent_id", "title", "description", "project", "tags"], row))
                r["tags"] = json.loads(r["tags"] or "[]")
                r["similarity"] = round(1 - (dist ** 2) / 2, 4)
                results.append(r)
        return results

    def count(self, project: str | None = None) -> int:
        if project:
            return self.conn.execute(
                "SELECT COUNT(*) FROM issues WHERE project = ?", (project,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
