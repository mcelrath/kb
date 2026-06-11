"""
Issues Repository

Manages issues (tasks, bugs, features, epics, etc.) for the kb-native tracker.
Modeled on ConceptRepository.

Parenthood source-of-truth: issues.parent_id column is the SOLE authority for
the parent-child relationship. Any 'parent-child' rows in issue_deps are
treated as redundant and are NOT used in the ready/blocked ancestor walk — the
recursive CTE walks issues.parent_id exclusively. This avoids double-counting
and keeps the logic simple: one FK, one CTE, no JOIN ambiguity.

ready/blocked blocking criteria:
  - A direct 'blocks' dep: issue B has row (B, A, 'blocks') meaning B depends_on A
    and A is still open/in_progress → B is blocked.
  - Parent blocked: if any ancestor via parent_id has an open 'blocks' blocker,
    the child is also blocked (children inherit parent's blocked state).
  - 'discovered-from', 'related', 'supersedes' dep types do NOT affect readiness.
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
DEP_TYPES = ("blocks", "parent-child", "discovered-from", "related", "supersedes")


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
        assignee: str | None = None,
        limit: int | None = None,
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
        if assignee:
            conditions.append("assignee = ?")
            params.append(assignee)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(limit)

        rows = self.conn.execute(
            f"""SELECT id, type, status, priority, parent_id, title, project, tags, created_at
                FROM issues {where} ORDER BY priority, created_at{limit_sql}""",
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

    # ------------------------------------------------------------------
    # Phase 2: dep management, comments, status transitions, ready/blocked, claim
    # ------------------------------------------------------------------

    def add_dep(
        self,
        issue_id: str,
        depends_on_id: str,
        dep_type: str,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Add a dependency edge: issue_id depends_on depends_on_id with dep_type.

        Uses INSERT OR IGNORE so duplicate calls are idempotent.
        dep_type must be one of DEP_TYPES (enforced by schema CHECK + this guard).
        Returns {"issue_id": ..., "depends_on_id": ..., "type": ..., "is_new": bool}.
        """
        if dep_type not in DEP_TYPES:
            raise ValueError(f"dep_type must be one of {DEP_TYPES}")
        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO issue_deps
               (issue_id, depends_on_id, type, created_at, created_by)
               VALUES (?, ?, ?, ?, ?)""",
            (issue_id, depends_on_id, dep_type, now, created_by),
        )
        self.conn.commit()
        return {
            "issue_id": issue_id,
            "depends_on_id": depends_on_id,
            "type": dep_type,
            "is_new": cur.rowcount == 1,
        }

    def list_deps(self, issue_id: str) -> dict[str, Any]:
        """Return outgoing and incoming deps for issue_id.

        Outgoing: this issue depends_on something (issue_id = issue_id).
        Incoming: something depends_on this issue (depends_on_id = issue_id).

        Each element carries: type, id (the other issue), title, status.
        Shape matches /dispatch's `dep list --json` consumer:
          {"outgoing": [{type, id, title, status}, ...],
           "incoming": [{type, id, title, status}, ...]}
        """
        outgoing_rows = self.conn.execute(
            """SELECT d.type, i.id, i.title, i.status
               FROM issue_deps d
               JOIN issues i ON i.id = d.depends_on_id
               WHERE d.issue_id = ?
               ORDER BY d.type, i.id""",
            (issue_id,),
        ).fetchall()

        incoming_rows = self.conn.execute(
            """SELECT d.type, i.id, i.title, i.status
               FROM issue_deps d
               JOIN issues i ON i.id = d.issue_id
               WHERE d.depends_on_id = ?
               ORDER BY d.type, i.id""",
            (issue_id,),
        ).fetchall()

        def _row_to_dict(r: tuple) -> dict:
            return {"type": r[0], "id": r[1], "title": r[2], "status": r[3]}

        return {
            "outgoing": [_row_to_dict(r) for r in outgoing_rows],
            "incoming": [_row_to_dict(r) for r in incoming_rows],
        }

    def add_comment(
        self,
        issue_id: str,
        body: str,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Add a comment to an issue.

        Comment id format: cmt-{utcnow}-{hex6}, mirroring the concepts id format.
        Returns {"id": cmt_id}.
        """
        now = datetime.utcnow().isoformat()
        cmt_id = f"cmt-{now}-{uuid.uuid4().hex[:6]}"
        self.conn.execute(
            "INSERT INTO issue_comments (id, issue_id, body, author, created_at) VALUES (?, ?, ?, ?, ?)",
            (cmt_id, issue_id, body, author, now),
        )
        self.conn.commit()
        return {"id": cmt_id}

    def set_status(
        self,
        issue_id: str,
        status: str,
        close_reason: str | None = None,
        closed_by_session: str | None = None,
    ) -> dict[str, Any]:
        """Update status on an issue, setting timestamp fields as appropriate.

        - open/in_progress: clears closed_at/close_reason.
        - closed: sets closed_at, close_reason, closed_by_session.
        - in_progress: sets started_at (if not already set).

        Returns {"id": issue_id, "status": new_status}.
        """
        if status not in ISSUE_STATUSES:
            raise ValueError(f"status must be one of {ISSUE_STATUSES}")
        now = datetime.utcnow().isoformat()

        if status == "closed":
            self.conn.execute(
                """UPDATE issues
                   SET status = ?, updated_at = ?, closed_at = ?,
                       close_reason = ?, closed_by_session = ?
                   WHERE id = ?""",
                (status, now, now, close_reason, closed_by_session, issue_id),
            )
        elif status == "in_progress":
            # Only set started_at if not already started (don't clobber reclaim).
            self.conn.execute(
                """UPDATE issues
                   SET status = ?, updated_at = ?,
                       started_at = COALESCE(started_at, ?),
                       closed_at = NULL, close_reason = NULL
                   WHERE id = ?""",
                (status, now, now, issue_id),
            )
        else:
            self.conn.execute(
                """UPDATE issues
                   SET status = ?, updated_at = ?,
                       closed_at = NULL, close_reason = NULL
                   WHERE id = ?""",
                (status, now, issue_id),
            )
        self.conn.commit()
        return {"id": issue_id, "status": status}

    def ready(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return issues that are ready to work on.

        An issue is READY iff:
          1. status IN ('open', 'in_progress')
          2. It has no direct 'blocks' blocker that is still open/in_progress:
             i.e. no row (this_issue, X, 'blocks') where X.status IN
             ('open','in_progress').
          3. None of its ancestors via parent_id has such a direct 'blocks' blocker.

        Parenthood source-of-truth: issues.parent_id column ONLY (not
        'parent-child' dep rows — those are redundant and NOT walked here).

        Implementation: WITH RECURSIVE CTE walks parent_id upward to collect all
        ancestor ids; then checks issue_deps for any open 'blocks' row against
        self OR any ancestor. Blocked issues are those with at least one open
        blocker; ready = not blocked.
        """
        conditions = "i.status IN ('open','in_progress')"
        params: list[Any] = []
        if project:
            conditions += " AND i.project = ?"
            params.append(project)

        sql = f"""
        WITH RECURSIVE ancestors(id, root_id) AS (
            -- Base: each issue is its own root
            SELECT id, id FROM issues
            UNION ALL
            -- Step: walk up via parent_id
            SELECT i.parent_id, a.root_id
            FROM issues i
            JOIN ancestors a ON a.id = i.id
            WHERE i.parent_id IS NOT NULL
        ),
        blocked_roots AS (
            -- A root_id is blocked if it OR any ancestor has an open 'blocks' dep
            SELECT DISTINCT a.root_id
            FROM ancestors a
            JOIN issue_deps d ON d.issue_id = a.id AND d.type = 'blocks'
            JOIN issues blocker ON blocker.id = d.depends_on_id
            WHERE blocker.status != 'closed'
        )
        SELECT i.id, i.type, i.status, i.priority, i.parent_id, i.title, i.project, i.tags, i.created_at
        FROM issues i
        WHERE {conditions}
          AND i.id NOT IN (SELECT root_id FROM blocked_roots)
        ORDER BY i.priority, i.created_at
        """
        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(zip(["id", "type", "status", "priority", "parent_id", "title", "project", "tags", "created_at"], r))
            d["tags"] = json.loads(d["tags"] or "[]")
            results.append(d)
        return results

    def blocked(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return issues that are blocked (have at least one open blocker).

        An issue is BLOCKED iff it is open/in_progress AND either:
          - it directly depends_on an open issue via 'blocks', OR
          - one of its ancestors (via parent_id) has such a blocker.

        Each result includes 'blocker_ids': list of the open blocking issue ids
        that directly caused the block (at self or ancestor level).

        Parenthood: issues.parent_id ONLY (not 'parent-child' dep rows).
        """
        conditions = "i.status IN ('open','in_progress')"
        params: list[Any] = []
        if project:
            conditions += " AND i.project = ?"
            params.append(project)

        sql = f"""
        WITH RECURSIVE ancestors(id, root_id) AS (
            SELECT id, id FROM issues
            UNION ALL
            SELECT i.parent_id, a.root_id
            FROM issues i
            JOIN ancestors a ON a.id = i.id
            WHERE i.parent_id IS NOT NULL
        ),
        blocker_edges AS (
            -- Collect (root_id, blocker_id) pairs: blocker is open, connected via 'blocks'
            SELECT a.root_id, d.depends_on_id AS blocker_id
            FROM ancestors a
            JOIN issue_deps d ON d.issue_id = a.id AND d.type = 'blocks'
            JOIN issues blocker ON blocker.id = d.depends_on_id
            WHERE blocker.status != 'closed'
        )
        SELECT i.id, i.type, i.status, i.priority, i.parent_id, i.title, i.project, i.tags, i.created_at,
               GROUP_CONCAT(be.blocker_id) AS blocker_ids
        FROM issues i
        JOIN blocker_edges be ON be.root_id = i.id
        WHERE {conditions}
        GROUP BY i.id
        ORDER BY i.priority, i.created_at
        """
        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(zip(["id", "type", "status", "priority", "parent_id", "title", "project", "tags", "created_at", "blocker_ids"], r))
            d["tags"] = json.loads(d["tags"] or "[]")
            # blocker_ids is a comma-separated string from GROUP_CONCAT; split it.
            raw = d["blocker_ids"]
            d["blocker_ids"] = list(dict.fromkeys(raw.split(",") if raw else []))
            results.append(d)
        return results

    def claim(self, issue_id: str, assignee: str) -> dict[str, Any]:
        """Atomically claim an issue for an assignee.

        Uses BEGIN IMMEDIATE + a single UPDATE with a WHERE guard (compare-and-swap):
          WHERE id = ? AND status IN ('open', 'blocked')
        rowcount == 1 → claimed successfully.
        rowcount == 0 → already claimed / closed / not found (not ours to take).

        PRAGMA busy_timeout is set so that a concurrent writer who holds the
        write lock causes SQLite to retry up to 5 s before raising OperationalError.
        An OperationalError "database is locked" means genuine write contention
        (SQLITE_BUSY after timeout), distinct from rowcount==0 (already claimed).

        Returns:
          {"claimed": True,  "id": issue_id}                        — success
          {"claimed": False, "already": True,  "id": issue_id}      — wrong status
          {"claimed": False, "contended": True, "id": issue_id}     — SQLITE_BUSY
        """
        self.conn.execute("PRAGMA busy_timeout = 5000")
        now = datetime.utcnow().isoformat()
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            cur = self.conn.execute(
                """UPDATE issues
                   SET status = 'in_progress', assignee = ?,
                       started_at = COALESCE(started_at, ?), updated_at = ?
                   WHERE id = ? AND status IN ('open', 'blocked')""",
                (assignee, now, now, issue_id),
            )
            self.conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                return {"claimed": False, "contended": True, "id": issue_id}
            raise
        if cur.rowcount == 1:
            return {"claimed": True, "id": issue_id}
        return {"claimed": False, "already": True, "id": issue_id}
