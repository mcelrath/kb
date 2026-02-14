"""
Scripts Repository

Manages script registration and retrieval.
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from .base import EntityRepository
from ..core.embedding import EmbeddingService


VALID_SCRIPT_RELATIONSHIPS = ("generated_by", "validates", "contradicts")


class ScriptsRepository(EntityRepository):
    """Repository for script management."""

    embedding_service: EmbeddingService
    finding_exists: Callable[[str], bool] | None

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_service: EmbeddingService,
        finding_exists: Callable[[str], bool] | None = None,
    ):
        super().__init__(conn)
        self.embedding_service = embedding_service
        self.finding_exists = finding_exists

    def add(
        self,
        path: str,
        purpose: str,
        project: str | None = None,
        language: str | None = None,
        store_content: bool = True,
        max_content_size: int = 100000,
    ) -> dict[str, object]:
        """Register a script in the knowledge base.

        Args:
            path: Path to the script file
            purpose: What hypothesis/question this script tests
            project: Project name
            language: Script language (auto-detected if not specified)
            store_content: Whether to store full script content
            max_content_size: Max content size to store (bytes)

        Returns:
            dict with 'id', 'is_new', 'filename'
        """
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Script not found: {path}")

        content = file_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for existing script with same hash
        existing = self.conn.execute(
            "SELECT id, filename FROM scripts WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing:
            return {"id": existing[0], "is_new": False, "filename": existing[1]}

        # Auto-detect language
        if language is None:
            suffix = file_path.suffix.lower()
            if suffix == ".py":
                language = "python"
            elif suffix == ".sage":
                language = "sage"
            elif suffix in (".sh", ".bash"):
                language = "bash"
            else:
                language = "other"

        # Only store content if within size limit
        stored_content = content if store_content and len(content) <= max_content_size else None

        script_id = f"script-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        now = datetime.now().isoformat()

        # Generate embedding BEFORE transaction
        embedding = self.embedding_service.embed(f"{file_path.name}: {purpose}")

        _ = self.conn.execute(
            """INSERT INTO scripts (id, path, filename, content_hash, content, purpose, project, language, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (script_id, str(file_path), file_path.name, content_hash, stored_content, purpose, project, language, now, now),
        )

        _ = self.conn.execute(
            "INSERT INTO scripts_vec (id, embedding) VALUES (?, ?)",
            (script_id, embedding),
        )

        self.conn.commit()
        return {"id": script_id, "is_new": True, "filename": file_path.name}

    def get(self, script_id: str) -> dict[str, object] | None:
        """Get a script by ID."""
        row = self.conn.execute(
            "SELECT * FROM scripts WHERE id = ?", (script_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "path": row[1],
            "filename": row[2],
            "content_hash": row[3],
            "content": row[4],
            "purpose": row[5],
            "project": row[6],
            "language": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }

    def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Search scripts by purpose using semantic similarity."""
        prefixed = f"Instruct: Given a search query, retrieve relevant research findings\nQuery: {query}"
        embedding = self.embedding_service.embed(prefixed)

        sql = """
            SELECT s.*, v.distance
            FROM scripts s
            JOIN scripts_vec v ON s.id = v.id
            WHERE v.embedding MATCH ?
              AND k = ?
        """
        params: list[object] = [embedding, limit * 2]

        if project:
            sql = sql.replace("WHERE", "WHERE s.project = ? AND")
            params = [project] + params

        sql += " ORDER BY v.distance"

        results: list[dict[str, object]] = []
        for row in self.conn.execute(sql, params).fetchall():
            script: dict[str, object] = {
                "id": row[0],
                "path": row[1],
                "filename": row[2],
                "content_hash": row[3],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "similarity": 1.0 - float(row[10]) / 2.0,
            }
            if project is None or script.get("project") == project:
                results.append(script)
                if len(results) >= limit:
                    break

        return results

    def list(
        self,
        project: str | None = None,
        language: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List registered scripts."""
        sql = "SELECT * FROM scripts WHERE 1=1"
        params: list[object] = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if language:
            sql += " AND language = ?"
            params.append(language)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results: list[dict[str, object]] = []
        for row in self.conn.execute(sql, params).fetchall():
            results.append({
                "id": row[0],
                "path": row[1],
                "filename": row[2],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "created_at": row[8],
            })
        return results

    def link_finding(
        self,
        finding_id: str,
        script_id: str,
        relationship: str = "generated_by",
    ) -> None:
        """Link a finding to a script that generated/validated it."""
        if relationship not in VALID_SCRIPT_RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship: {relationship}. "
                f"Must be one of {VALID_SCRIPT_RELATIONSHIPS}"
            )

        # Verify both exist
        if self.finding_exists and not self.finding_exists(finding_id):
            raise ValueError(f"Finding not found: {finding_id}")
        if not self.get(script_id):
            raise ValueError(f"Script not found: {script_id}")

        now = datetime.now().isoformat()
        _ = self.conn.execute(
            """INSERT OR REPLACE INTO finding_scripts (finding_id, script_id, relationship, linked_at)
               VALUES (?, ?, ?, ?)""",
            (finding_id, script_id, relationship, now),
        )
        self.conn.commit()

    def get_findings(self, script_id: str) -> list[dict[str, object]]:
        """Get findings generated by a script."""
        rows = self.conn.execute(
            """SELECT f.*, fs.relationship
               FROM findings f
               JOIN finding_scripts fs ON f.id = fs.finding_id
               WHERE fs.script_id = ?
               ORDER BY f.created_at DESC""",
            (script_id,),
        ).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            results.append({
                "id": row[0],
                "type": row[1],
                "content": row[7],
                "relationship": row[11],
            })
        return results

    def get_for_finding(self, finding_id: str) -> list[dict[str, object]]:
        """Get scripts that generated a finding."""
        rows = self.conn.execute(
            """SELECT s.*, fs.relationship
               FROM scripts s
               JOIN finding_scripts fs ON s.id = fs.script_id
               WHERE fs.finding_id = ?""",
            (finding_id,),
        ).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            results.append({
                "id": row[0],
                "filename": row[2],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "relationship": row[10],
            })
        return results

    def delete(self, script_id: str) -> bool:
        """Delete a script and its links."""
        _ = self.conn.execute("DELETE FROM finding_scripts WHERE script_id = ?", (script_id,))
        _ = self.conn.execute("DELETE FROM scripts_vec WHERE id = ?", (script_id,))
        cursor = self.conn.execute("DELETE FROM scripts WHERE id = ?", (script_id,))
        self.conn.commit()
        return cursor.rowcount > 0
