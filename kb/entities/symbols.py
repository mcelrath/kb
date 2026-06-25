"""
Symbols Repository

Manages the code-symbol index (any language) — add, prune, delete, search.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService


_LANG_BY_EXT = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".rs": "rust", ".go": "go", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".java": "java", ".rb": "ruby",
    ".tex": "tex", ".lean": "lean",
}

import os as _os


def _lang_from_file(path: str | None) -> str | None:
    """Derive a symbol's language from its file extension."""
    return _LANG_BY_EXT.get(_os.path.splitext(path or "")[1].lower())


class SymbolsRepository(EntityRepository):
    """Repository for code-symbol indexing (any language)."""

    embedding_service: EmbeddingService

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self.embedding_service = embedding_service

    @staticmethod
    def _python_symbol_content_hash(
        project: str | None,
        module: str,
        name: str,
        signature: str,
        docstring_summary: str | None,
    ) -> str:
        """SHA-256 content hash for a Python symbol.

        Covers project+module+name (identity) plus signature and
        docstring_summary (content), so decorator/return-type changes and
        docstring edits are detected as distinct.
        """
        import hashlib
        raw = "\x00".join([
            project or "",
            module,
            name,
            signature,
            docstring_summary or "",
        ])
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _python_symbol_stable_id(project: str | None, module: str, name: str) -> str:
        """Stable deterministic ID for a Python symbol, keyed on (project, module, name).

        This is the *identity* hash -- it does not change when the signature or
        docstring changes, so it can be used as a stable FK / lookup key.
        """
        import hashlib
        raw = "\x00".join([project or "", module, name])
        return "pysym-" + hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _ensure_python_symbol_hash_columns(self) -> None:
        """Add content_hash and symbol_id columns if they don't exist yet (idempotent)."""
        existing_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        if "content_hash" not in existing_cols:
            self.conn.execute(
                "ALTER TABLE symbols ADD COLUMN content_hash TEXT"
            )
        if "symbol_id" not in existing_cols:
            self.conn.execute(
                "ALTER TABLE symbols ADD COLUMN symbol_id TEXT"
            )
        self.conn.commit()

    def add_symbol(
        self,
        name: str,
        kind: str,
        module: str,
        signature: str,
        file: str,
        line: int,
        status: str = "public",
        is_lru_cached: bool = False,
        frame_hint: str | None = None,
        redirect_to: str | None = None,
        docstring_summary: str | None = None,
        lean_citations: list[str] | None = None,
        kb_refs: list[str] | None = None,
        also_in_modules: list[dict[str, Any]] | None = None,
        project: str | None = None,
        parent_impl: str | None = None,
        visibility: str | None = None,
        is_signature_only: bool = False,
        node_type: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Add or update a symbol in the index (any language).

        Returns dict with 'id', 'is_new'.
        """
        # Ensure new columns exist (idempotent ALTER TABLE; fast after first call)
        self._ensure_python_symbol_hash_columns()

        content_hash = self._python_symbol_content_hash(
            project, module, name, signature, docstring_summary
        )
        symbol_id = self._python_symbol_stable_id(project, module, name)

        existing = self.conn.execute(
            "SELECT id, content_hash "
            "FROM symbols WHERE name = ? AND module = ?",
            (name, module),
        ).fetchone()

        now = datetime.now().isoformat()
        lang = language or _lang_from_file(file)
        lean_json = json.dumps(lean_citations or [])
        kb_json = json.dumps(kb_refs or [])
        also_json = json.dumps(also_in_modules or [])

        if existing:
            # Skip embedding + UPDATE entirely when nothing material changed.
            if existing["content_hash"] == content_hash:
                return {"id": existing["id"], "is_new": False, "skipped": True}

        embed_text = f"{module}.{name}: {signature} {docstring_summary or ''}"
        embedding = self.embedding_service.embed(embed_text)

        if existing:
            sym_id = existing["id"]
            self.conn.execute("""
                UPDATE symbols SET kind=?, signature=?, status=?, is_lru_cached=?,
                    frame_hint=?, redirect_to=?, docstring_summary=?, lean_citations=?,
                    kb_refs=?, also_in_modules=?, file=?, line=?, project=?,
                    updated_at=?, embedding=?, content_hash=?, symbol_id=?,
                    parent_impl=?, visibility=?, is_signature_only=?, node_type=?, language=?
                WHERE id=?
            """, (kind, signature, status, int(is_lru_cached), frame_hint, redirect_to,
                  docstring_summary, lean_json, kb_json, also_json, file, line, project,
                  now, embedding, content_hash, symbol_id,
                  parent_impl, visibility, int(is_signature_only), node_type, lang,
                  sym_id))
            self.conn.execute("DELETE FROM symbols_vec WHERE id = ?", (sym_id,))
            self.conn.execute(
                "INSERT INTO symbols_vec (id, embedding) VALUES (?, ?)",
                (sym_id, embedding),
            )
            self.conn.commit()
            return {"id": sym_id, "is_new": False}

        sym_id = symbol_id
        self.conn.execute("""
            INSERT INTO symbols
                (id, name, kind, module, signature, status, is_lru_cached,
                 frame_hint, redirect_to, docstring_summary, lean_citations,
                 kb_refs, also_in_modules, file, line, project, created_at, updated_at,
                 embedding, content_hash, symbol_id,
                 parent_impl, visibility, is_signature_only, node_type, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
        """, (sym_id, name, kind, module, signature, status, int(is_lru_cached),
              frame_hint, redirect_to, docstring_summary, lean_json, kb_json, also_json,
              file, line, project, now, now, embedding, content_hash, symbol_id,
              parent_impl, visibility, int(is_signature_only), node_type, lang))
        self.conn.execute(
            "INSERT INTO symbols_vec (id, embedding) VALUES (?, ?)",
            (sym_id, embedding),
        )
        self.conn.commit()
        return {
            "id": sym_id,
            "is_new": True,
            "parent_impl": parent_impl,
            "visibility": visibility,
            "is_signature_only": is_signature_only,
            "node_type": node_type,
        }

    def prune_symbols_for_file(
        self,
        file: str,
        live_names_modules: set[tuple[str, str]],
    ) -> int:
        """Delete stale symbols rows for a file after re-ingest.

        Removes rows whose (name, module) is NOT in live_names_modules.
        Also cleans symbols_vec.  Returns count of deleted rows.

        Guard: if live_names_modules is empty, nothing is deleted (parse
        failure / empty file must not wipe existing rows).
        """
        if not live_names_modules:
            return 0
        rows = self.conn.execute(
            "SELECT id, name, module FROM symbols WHERE file = ?",
            (file,),
        ).fetchall()
        to_delete = [
            row["id"]
            for row in rows
            if (row["name"], row["module"]) not in live_names_modules
        ]
        if not to_delete:
            return 0
        for sid in to_delete:
            self.conn.execute("DELETE FROM symbols_vec WHERE id = ?", (sid,))
            self.conn.execute("DELETE FROM symbols WHERE id = ?", (sid,))
        self.conn.commit()
        return len(to_delete)

    def delete_symbols_for_file(self, file: str) -> int:
        """Remove ALL symbols rows for a deleted/removed file.

        Also cleans symbols_vec. Returns count of deleted rows.
        """
        rows = self.conn.execute(
            "SELECT id FROM symbols WHERE file = ?", (file,)
        ).fetchall()
        for row in rows:
            self.conn.execute("DELETE FROM symbols_vec WHERE id = ?", (row["id"],))
        result = self.conn.execute(
            "DELETE FROM symbols WHERE file = ?", (file,)
        )
        self.conn.commit()
        return result.rowcount

    def search_symbols(
        self,
        query: str,
        module: str | None = None,
        status: str | None = None,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search Python symbols by semantic similarity."""
        embedding = self.embedding_service.embed(query)

        conditions = []
        params: list[Any] = []
        if module:
            conditions.append("p.module LIKE ?")
            params.append(f"{module}%")
        if status:
            conditions.append("p.status = ?")
            params.append(status)
        if project:
            conditions.append("p.project = ?")
            params.append(project)

        vec_results = self.conn.execute(
            f"""SELECT v.id, v.distance
                FROM symbols_vec v
                JOIN symbols p ON p.id = v.id
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
                """SELECT id, name, kind, module, signature, status, frame_hint,
                          docstring_summary, lean_citations, kb_refs, file, line, project
                   FROM symbols WHERE id = ?""",
                (sid,),
            ).fetchone()
            if row:
                r = dict(zip([
                    "id", "name", "kind", "module", "signature", "status", "frame_hint",
                    "docstring_summary", "lean_citations", "kb_refs", "file", "line", "project"
                ], row))
                r["similarity"] = round(seen[sid], 4)
                for fld in ("lean_citations", "kb_refs"):
                    if r.get(fld):
                        try:
                            r[fld] = json.loads(r[fld])
                        except json.JSONDecodeError:
                            r[fld] = []
                results.append(r)
        return results
