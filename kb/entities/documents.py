"""
Documents Repository

Manages authoritative documents and citations.
"""

import os
import sqlite3
from datetime import datetime

from .base import EntityRepository


VALID_DOC_TYPES = ("spec", "paper", "standard", "internal", "reference")
VALID_CITATION_TYPES = ("references", "implements", "contradicts", "extends")


class DocumentsRepository(EntityRepository):
    """Repository for document management."""

    def add(
        self,
        title: str,
        doc_type: str,
        url: str | None = None,
        project: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Add an authoritative document.

        doc_type: spec, paper, standard, internal, reference
        """
        if doc_type not in VALID_DOC_TYPES:
            raise ValueError(f"Invalid doc_type: {doc_type}. Must be one of {VALID_DOC_TYPES}")

        now = datetime.now().isoformat()
        doc_id = f"doc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"

        _ = self.conn.execute(
            """INSERT INTO documents (id, title, url, doc_type, project, summary, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?)""",
            (doc_id, title, url, doc_type, project, summary, now)
        )
        self.conn.commit()
        return doc_id

    def get(self, doc_id: str) -> dict[str, object] | None:
        """Get a document by ID with citation count."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if not row:
            return None

        # Count citations
        citation_count = self.conn.execute(
            "SELECT COUNT(*) FROM document_citations WHERE document_id = ?",
            (doc_id,)
        ).fetchone()[0]

        return {
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "doc_type": row["doc_type"],
            "project": row["project"],
            "status": row["status"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "superseded_by": row["superseded_by"],
            "citation_count": citation_count,
        }

    def list(
        self,
        project: str | None = None,
        doc_type: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List documents with optional filters."""
        sql = "SELECT * FROM documents WHERE 1=1"
        params: list[object] = []

        if not include_superseded:
            sql += " AND status = 'active'"
        if project:
            sql += " AND project = ?"
            params.append(project)
        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "doc_type": row["doc_type"],
                "project": row["project"],
                "status": row["status"],
                "summary": row["summary"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def search(self, query: str, project: str | None = None) -> list[dict[str, object]]:
        """Search documents by title or summary."""
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        sql = """SELECT * FROM documents
                 WHERE (title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\')
                 AND status = 'active'"""
        params: list[object] = [pattern, pattern]

        if project:
            sql += " AND project = ?"
            params.append(project)

        sql += " ORDER BY created_at DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "doc_type": row["doc_type"],
                "project": row["project"],
                "summary": row["summary"],
            }
            for row in rows
        ]

    def supersede(self, doc_id: str, new_doc_id: str) -> bool:
        """Mark a document as superseded by another."""
        cursor = self.conn.execute(
            "UPDATE documents SET status = 'superseded', superseded_by = ? WHERE id = ?",
            (new_doc_id, doc_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def cite(
        self,
        finding_id: str,
        doc_id: str,
        citation_type: str = "references",
        notes: str | None = None,
    ) -> bool:
        """Link a finding to a document it cites.

        citation_type: references, implements, contradicts, extends
        """
        if citation_type not in VALID_CITATION_TYPES:
            raise ValueError(f"Invalid citation_type: {citation_type}. Must be one of {VALID_CITATION_TYPES}")

        now = datetime.now().isoformat()

        try:
            _ = self.conn.execute(
                """INSERT INTO document_citations (finding_id, document_id, citation_type, notes, cited_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (finding_id, doc_id, citation_type, notes, now)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def get_citations(self, doc_id: str) -> list[dict[str, object]]:
        """Get all findings that cite a document."""
        rows = self.conn.execute(
            """SELECT dc.*, f.content, f.type, f.project
               FROM document_citations dc
               JOIN findings f ON dc.finding_id = f.id
               WHERE dc.document_id = ?
               ORDER BY dc.cited_at DESC""",
            (doc_id,)
        ).fetchall()

        return [
            {
                "finding_id": row["finding_id"],
                "content": row["content"],
                "type": row["type"],
                "project": row["project"],
                "citation_type": row["citation_type"],
                "notes": row["notes"],
                "cited_at": row["cited_at"],
            }
            for row in rows
        ]

    def get_docs_for_finding(self, finding_id: str) -> list[dict[str, object]]:
        """Get all documents cited by a finding."""
        rows = self.conn.execute(
            """SELECT d.*, dc.citation_type, dc.notes, dc.cited_at
               FROM documents d
               JOIN document_citations dc ON d.id = dc.document_id
               WHERE dc.finding_id = ?
               ORDER BY dc.cited_at DESC""",
            (finding_id,)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "doc_type": row["doc_type"],
                "citation_type": row["citation_type"],
                "notes": row["notes"],
                "cited_at": row["cited_at"],
            }
            for row in rows
        ]

    def delete(self, doc_id: str) -> bool:
        """Delete a document and its citations."""
        _ = self.conn.execute("DELETE FROM document_citations WHERE document_id = ?", (doc_id,))
        cursor = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0
