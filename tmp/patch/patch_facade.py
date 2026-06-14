"""Patch kb/facade.py: add content_hash + symbol_id + prune/delete methods."""

with open('/home/mcelrath/Projects/ai/kb/kb/facade.py', 'r') as f:
    content = f.read()

START_MARKER = '    def add_python_symbol(\n'
END_MARKER = '        self.conn.commit()\n        return {"id": sym_id, "is_new": True}\n'

start_idx = content.find(START_MARKER)
assert start_idx != -1, "start not found"
end_idx = content.find(END_MARKER, start_idx)
assert end_idx != -1, "end not found"
end_idx += len(END_MARKER)

old_block = content[start_idx:end_idx]
print(f"Old block: {len(old_block)} chars, starts at char {start_idx}")

NEW_BLOCK = '''\
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
        raw = "\\x00".join([
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
        raw = "\\x00".join([project or "", module, name])
        return "pysym-" + hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _ensure_python_symbol_hash_columns(self) -> None:
        """Add content_hash and symbol_id columns if they don\'t exist yet (idempotent)."""
        existing_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(python_symbols)").fetchall()
        }
        if "content_hash" not in existing_cols:
            self.conn.execute(
                "ALTER TABLE python_symbols ADD COLUMN content_hash TEXT"
            )
        if "symbol_id" not in existing_cols:
            self.conn.execute(
                "ALTER TABLE python_symbols ADD COLUMN symbol_id TEXT"
            )
        self.conn.commit()

    def add_python_symbol(
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
    ) -> dict[str, Any]:
        """Add or update a Python symbol in the index.

        Returns dict with \'id\', \'is_new\'.
        """
        # Ensure new columns exist (idempotent ALTER TABLE; fast after first call)
        self._ensure_python_symbol_hash_columns()

        content_hash = self._python_symbol_content_hash(
            project, module, name, signature, docstring_summary
        )
        symbol_id = self._python_symbol_stable_id(project, module, name)

        existing = self.conn.execute(
            "SELECT id, content_hash "
            "FROM python_symbols WHERE name = ? AND module = ?",
            (name, module),
        ).fetchone()

        now = datetime.now().isoformat()
        lean_json = json.dumps(lean_citations or [])
        kb_json = json.dumps(kb_refs or [])
        also_json = json.dumps(also_in_modules or [])

        if existing:
            # Skip embedding + UPDATE entirely when nothing material changed.
            if existing["content_hash"] == content_hash:
                return {"id": existing["id"], "is_new": False, "skipped": True}

        embed_text = f"{module}.{name}: {signature} {docstring_summary or \'\'}"
        embedding = self._embed(embed_text)

        if existing:
            sym_id = existing["id"]
            self.conn.execute("""
                UPDATE python_symbols SET kind=?, signature=?, status=?, is_lru_cached=?,
                    frame_hint=?, redirect_to=?, docstring_summary=?, lean_citations=?,
                    kb_refs=?, also_in_modules=?, file=?, line=?, project=?,
                    updated_at=?, embedding=?, content_hash=?, symbol_id=?
                WHERE id=?
            """, (kind, signature, status, int(is_lru_cached), frame_hint, redirect_to,
                  docstring_summary, lean_json, kb_json, also_json, file, line, project,
                  now, embedding, content_hash, symbol_id, sym_id))
            self.conn.execute("DELETE FROM python_symbols_vec WHERE id = ?", (sym_id,))
            self.conn.execute(
                "INSERT INTO python_symbols_vec (id, embedding) VALUES (?, ?)",
                (sym_id, embedding),
            )
            self.conn.commit()
            return {"id": sym_id, "is_new": False}

        sym_id = symbol_id
        self.conn.execute("""
            INSERT INTO python_symbols
                (id, name, kind, module, signature, status, is_lru_cached,
                 frame_hint, redirect_to, docstring_summary, lean_citations,
                 kb_refs, also_in_modules, file, line, project, created_at, updated_at,
                 embedding, content_hash, symbol_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sym_id, name, kind, module, signature, status, int(is_lru_cached),
              frame_hint, redirect_to, docstring_summary, lean_json, kb_json, also_json,
              file, line, project, now, now, embedding, content_hash, symbol_id))
        self.conn.execute(
            "INSERT INTO python_symbols_vec (id, embedding) VALUES (?, ?)",
            (sym_id, embedding),
        )
        self.conn.commit()
        return {"id": sym_id, "is_new": True}

    def prune_python_symbols_for_file(
        self,
        file: str,
        live_names_modules: set[tuple[str, str]],
    ) -> int:
        """Delete stale python_symbols rows for a file after re-ingest.

        Removes rows whose (name, module) is NOT in live_names_modules.
        Also cleans python_symbols_vec.  Returns count of deleted rows.

        Guard: if live_names_modules is empty, nothing is deleted (parse
        failure / empty file must not wipe existing rows).
        """
        if not live_names_modules:
            return 0
        rows = self.conn.execute(
            "SELECT id, name, module FROM python_symbols WHERE file = ?",
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
            self.conn.execute("DELETE FROM python_symbols_vec WHERE id = ?", (sid,))
            self.conn.execute("DELETE FROM python_symbols WHERE id = ?", (sid,))
        self.conn.commit()
        return len(to_delete)

    def delete_python_symbols_for_file(self, file: str) -> int:
        """Remove ALL python_symbols rows for a deleted/removed file.

        Also cleans python_symbols_vec. Returns count of deleted rows.
        """
        rows = self.conn.execute(
            "SELECT id FROM python_symbols WHERE file = ?", (file,)
        ).fetchall()
        for row in rows:
            self.conn.execute("DELETE FROM python_symbols_vec WHERE id = ?", (row["id"],))
        result = self.conn.execute(
            "DELETE FROM python_symbols WHERE file = ?", (file,)
        )
        self.conn.commit()
        return result.rowcount
'''

new_content = content[:start_idx] + NEW_BLOCK + content[end_idx:]
assert new_content != content, "No change made"

with open('/home/mcelrath/Projects/ai/kb/kb/facade.py', 'w') as f:
    f.write(new_content)

print(f"Done. Lines: {new_content.count(chr(10))}")
