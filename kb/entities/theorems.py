"""
Lean Theorem Repository

Manages Lean theorem declarations with vector search.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService
from ..validation import serialize_f32


class TheoremRepository(EntityRepository):
    """Repository for Lean theorem storage and retrieval."""

    embedding_service: EmbeddingService

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self.embedding_service = embedding_service

    def add(
        self,
        lean_name: str,
        name: str,
        statement: str,
        declaration: str,
        file: str,
        statement_pure: str | None = None,
        module: str | None = None,
        line: int | None = None,
        tex_source: str | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a Lean theorem to the index.

        Returns dict with 'id', 'is_new'.
        """
        existing = self.conn.execute(
            "SELECT id, finding_id FROM lean_theorems WHERE lean_name = ? AND file = ?",
            (lean_name, file),
        ).fetchone()
        if existing:
            # Backfill finding_id if still NULL
            if existing[1] is None:
                lean_name_base = lean_name.split("::")[-1] if "::" in lean_name else lean_name
                matching_finding = self.conn.execute(
                    """SELECT id FROM findings
                       WHERE content LIKE ? OR tags LIKE ?
                       LIMIT 1""",
                    (f"%lean:{lean_name_base}%", f"%lean:{lean_name_base}%"),
                ).fetchone()
                if matching_finding:
                    self.conn.execute(
                        "UPDATE lean_theorems SET finding_id = ? WHERE id = ?",
                        (matching_finding[0], existing[0]),
                    )
                    self.conn.commit()
            return {"id": existing[0], "is_new": False}

        tid = f"thm-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.utcnow().isoformat()
        tags_json = json.dumps(tags) if tags else None

        self.conn.execute(
            """INSERT INTO lean_theorems
               (id, lean_name, name, statement, statement_pure, declaration, module,
                file, line, tex_source, project, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tid, lean_name, name, statement, statement_pure, declaration, module,
             file, line, tex_source, project, tags_json, now, now),
        )

        embed_text = statement_pure if statement_pure else statement
        embedding = self.embedding_service.embed(embed_text)
        self.conn.execute("DELETE FROM lean_theorems_vec WHERE id = ?", (tid,))
        self.conn.execute(
            "INSERT INTO lean_theorems_vec (id, embedding) VALUES (?, ?)",
            (tid, embedding),
        )

        # Backfill finding_id: check if any finding already cites this theorem by name
        lean_name_base = lean_name.split("::")[-1] if "::" in lean_name else lean_name
        matching_finding = self.conn.execute(
            """SELECT id FROM findings
               WHERE content LIKE ? OR tags LIKE ?
               LIMIT 1""",
            (f"%lean:{lean_name_base}%", f"%{lean_name_base}%"),
        ).fetchone()
        if matching_finding:
            self.conn.execute(
                "UPDATE lean_theorems SET finding_id = ? WHERE id = ?",
                (matching_finding[0], tid),
            )

        self.conn.commit()
        return {"id": tid, "is_new": True}

    def get(self, theorem_id: str) -> dict[str, Any] | None:
        """Get a theorem by ID."""
        row = self.conn.execute(
            "SELECT * FROM lean_theorems WHERE id = ?", (theorem_id,)
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM lean_theorems LIMIT 0").description]
        result = dict(zip(cols, row))
        if result.get("tags"):
            result["tags"] = json.loads(result["tags"])
        deps = self.conn.execute(
            "SELECT depends_on_id FROM theorem_dependencies WHERE theorem_id = ?",
            (theorem_id,),
        ).fetchall()
        result["dependencies"] = [r[0] for r in deps]
        return result

    def search(
        self,
        query: str,
        module: str | None = None,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search theorems by semantic similarity + FTS."""
        embedding = self.embedding_service.embed(query)

        conditions = []
        params: list[Any] = []
        if module:
            conditions.append("t.module LIKE ?")
            params.append(f"{module}%")
        if project:
            conditions.append("t.project = ?")
            params.append(project)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        vec_results = []
        if embedding is not None:
            vec_results = self.conn.execute(
                f"""SELECT v.id, v.distance
                    FROM lean_theorems_vec v
                    JOIN lean_theorems t ON t.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                    {"AND " + " AND ".join(conditions) if conditions else ""}
                    ORDER BY v.distance""",
                (embedding, limit * 2, *params),
            ).fetchall()

        fts_and = ("AND " + " AND ".join(conditions)) if conditions else ""
        fts_results = self.conn.execute(
            f"""SELECT t.id, rank
                FROM lean_theorems_fts f
                JOIN lean_theorems t ON t.rowid = f.rowid
                WHERE lean_theorems_fts MATCH ?
                {fts_and}
                LIMIT ?""",
            (query, *params, limit),
        ).fetchall() if query.strip() else []

        seen: dict[str, float] = {}
        for tid, dist in vec_results:
            if dist is not None:
                seen[tid] = 1 - (dist ** 2) / 2
        for tid, rank in fts_results:
            if tid not in seen:
                seen[tid] = 0.6

        top_ids = sorted(seen, key=lambda x: seen[x], reverse=True)[:limit]
        results = []
        for tid in top_ids:
            row = self.conn.execute(
                "SELECT id, lean_name, name, statement, statement_pure, module, file, line, project, tags FROM lean_theorems WHERE id = ?",
                (tid,),
            ).fetchone()
            if row:
                r = dict(zip(["id","lean_name","name","statement","statement_pure","module","file","line","project","tags"], row))
                r["similarity"] = round(seen[tid], 4)
                if r.get("tags"):
                    r["tags"] = json.loads(r["tags"])
                results.append(r)
        return results

    def search_by_tex_source(self, tex_ref: str) -> list[dict[str, Any]]:
        """Find theorems by tex_source cross-reference."""
        rows = self.conn.execute(
            "SELECT id, lean_name, name, statement, file, line FROM lean_theorems WHERE tex_source LIKE ?",
            (f"%{tex_ref}%",),
        ).fetchall()
        return [dict(zip(["id","lean_name","name","statement","file","line"], r)) for r in rows]

    def list_module(self, module_path: str) -> list[dict[str, Any]]:
        """List all theorems in a module or submodule."""
        rows = self.conn.execute(
            "SELECT id, lean_name, name, statement_pure, statement, file, line FROM lean_theorems WHERE module LIKE ? ORDER BY file, line",
            (f"{module_path}%",),
        ).fetchall()
        return [dict(zip(["id","lean_name","name","statement_pure","statement","file","line"], r)) for r in rows]

    def add_dependency(self, theorem_id: str, depends_on_id: str) -> None:
        """Record that theorem_id depends on depends_on_id."""
        self.conn.execute(
            "INSERT OR IGNORE INTO theorem_dependencies (theorem_id, depends_on_id) VALUES (?, ?)",
            (theorem_id, depends_on_id),
        )
        self.conn.commit()

    def get_dependencies(self, theorem_id: str) -> list[dict[str, Any]]:
        """Get theorems that this theorem depends on."""
        rows = self.conn.execute(
            """SELECT t.id, t.lean_name, t.name, t.statement_pure, t.statement
               FROM theorem_dependencies d
               JOIN lean_theorems t ON t.id = d.depends_on_id
               WHERE d.theorem_id = ?""",
            (theorem_id,),
        ).fetchall()
        return [dict(zip(["id","lean_name","name","statement_pure","statement"], r)) for r in rows]

    def update_statement_pure(self, theorem_id: str, statement_pure: str) -> None:
        """Update the pure-math restatement and re-embed."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE lean_theorems SET statement_pure = ?, updated_at = ? WHERE id = ?",
            (statement_pure, now, theorem_id),
        )
        embedding = self.embedding_service.embed(statement_pure)
        self.conn.execute("DELETE FROM lean_theorems_vec WHERE id = ?", (theorem_id,))
        self.conn.execute(
            "INSERT INTO lean_theorems_vec (id, embedding) VALUES (?, ?)",
            (theorem_id, embedding),
        )
        self.conn.commit()

    def count(self, project: str | None = None) -> int:
        if project:
            return self.conn.execute(
                "SELECT COUNT(*) FROM lean_theorems WHERE project = ?", (project,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM lean_theorems").fetchone()[0]

    def fetch_missing_statement_pure(
        self, project: str | None = None, limit: int | None = None
    ) -> list[Any]:
        """Return (id, lean_name, statement) rows that lack a statement_pure."""
        where = "WHERE statement_pure IS NULL OR statement_pure = ''"
        params: list[Any] = []
        if project:
            where += " AND project = ?"
            params.append(project)
        sql = f"SELECT id, lean_name, statement FROM lean_theorems {where}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.conn.execute(sql, params).fetchall()

    def set_statement_pure(self, theorem_id: str, pure: str) -> None:
        """Write a statement_pure value (does NOT re-embed; call reembed_statement_pure after)."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE lean_theorems SET statement_pure = ?, updated_at = ? WHERE id = ?",
            (pure, now, theorem_id),
        )
        # Caller is responsible for commit batching.

    def reembed_statement_pure(self, theorem_id: str, pure: str) -> None:
        """Delete+insert the vec row for a single theorem using statement_pure text."""
        embedding = self.embedding_service.embed(pure)
        self.conn.execute("DELETE FROM lean_theorems_vec WHERE id = ?", (theorem_id,))
        self.conn.execute(
            "INSERT INTO lean_theorems_vec (id, embedding) VALUES (?, ?)",
            (theorem_id, embedding),
        )
        # Caller is responsible for commit batching.
