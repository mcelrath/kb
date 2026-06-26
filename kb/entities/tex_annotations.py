"""TeX annotation repository — physics/Lean cross-reference index."""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService


class TexAnnotationsRepository(EntityRepository):
    """Repository for TeX ↔ Python/Lean cross-reference annotations."""

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self._embedding = embedding_service

    def add(
        self,
        file: str,
        line: int,
        section_label: str | None = None,
        section_title: str | None = None,
        python_refs: list[str] | None = None,
        lean_refs: list[str] | None = None,
        epic_refs: list[str] | None = None,
        kb_refs: list[str] | None = None,
        context: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Add or update a TeX annotation. Returns dict with 'id', 'is_new'."""
        existing = self.conn.execute(
            "SELECT id FROM tex_annotations WHERE file = ? AND line = ?",
            (file, line),
        ).fetchone()

        now = datetime.now().isoformat()
        ann_id = f"texann-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        embed_text = " ".join(filter(None, [
            section_title or "",
            section_label or "",
            f"python:{json.dumps(python_refs or [])}",
            f"lean:{json.dumps(lean_refs or [])}",
            context or "",
        ]))
        embedding = self._embedding.embed(embed_text)

        python_json = json.dumps(python_refs or [])
        lean_json = json.dumps(lean_refs or [])
        epic_json = json.dumps(epic_refs or [])
        kb_json = json.dumps(kb_refs or [])

        if existing:
            ann_id = existing[0]
            self.conn.execute("""
                UPDATE tex_annotations SET section_label=?, section_title=?,
                    python_refs=?, lean_refs=?, epic_refs=?, kb_refs=?,
                    context=?, updated_at=?, embedding=?
                WHERE id=?
            """, (section_label, section_title, python_json, lean_json, epic_json,
                  kb_json, context, now, embedding, ann_id))
            self.conn.execute("DELETE FROM tex_annotations_vec WHERE id = ?", (ann_id,))
            self.conn.execute(
                "INSERT INTO tex_annotations_vec (id, embedding) VALUES (?, ?)",
                (ann_id, embedding),
            )
            self.conn.commit()
            return {"id": ann_id, "is_new": False}

        self.conn.execute("""
            INSERT INTO tex_annotations
                (id, section_label, section_title, python_refs, lean_refs, epic_refs,
                 kb_refs, context, file, line, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ann_id, section_label, section_title, python_json, lean_json, epic_json,
              kb_json, context, file, line, now, now, embedding))
        self.conn.execute(
            "INSERT INTO tex_annotations_vec (id, embedding) VALUES (?, ?)",
            (ann_id, embedding),
        )
        self.conn.commit()
        return {"id": ann_id, "is_new": True}

    def search(
        self,
        query: str,
        file: str | None = None,
        section_label: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search TeX annotations by semantic similarity."""
        embedding = self._embedding.embed(query)

        conditions = []
        params: list[Any] = []
        if file:
            conditions.append("t.file = ?")
            params.append(file)
        if section_label:
            conditions.append("t.section_label = ?")
            params.append(section_label)

        vec_results = self.conn.execute(
            f"""SELECT v.id, v.distance
                FROM tex_annotations_vec v
                JOIN tex_annotations t ON t.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                {"AND " + " AND ".join(conditions) if conditions else ""}
                ORDER BY v.distance""",
            (embedding, limit * 2, *params),
        ).fetchall()

        seen: dict[str, float] = {}
        for sid, dist in vec_results:
            if dist is not None:
                seen[sid] = 1 - (dist ** 2) / 2

        top_ids = sorted(seen, key=lambda x: seen[x], reverse=True)[:limit]
        results = []
        for sid in top_ids:
            row = self.conn.execute(
                """SELECT id, section_label, section_title, python_refs, lean_refs,
                          epic_refs, kb_refs, context, file, line
                   FROM tex_annotations WHERE id = ?""",
                (sid,),
            ).fetchone()
            if row:
                r = dict(zip([
                    "id", "section_label", "section_title", "python_refs", "lean_refs",
                    "epic_refs", "kb_refs", "context", "file", "line"
                ], row))
                r["similarity"] = round(seen[sid], 4)
                for fld in ("python_refs", "lean_refs", "epic_refs", "kb_refs"):
                    if r.get(fld):
                        try:
                            r[fld] = json.loads(r[fld])
                        except json.JSONDecodeError:
                            r[fld] = []
                results.append(r)
        return results
