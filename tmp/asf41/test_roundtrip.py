"""
Scratch test: verify that parent_impl/visibility/is_signature_only/node_type
round-trip through add_python_symbol via a temp SQLite DB.

Run with: .venv/bin/python tmp/asf41/test_roundtrip.py
"""

import sqlite3
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kb.core.schema import init_schema

# Minimal stub for EmbeddingService that returns a fixed zero-vector blob
EMBEDDING_DIM = 4


def fake_embed(text: str) -> bytes:
    """Return a normalized dummy embedding (4-dim) as float32 bytes."""
    import struct
    # Unit vector along first axis
    vals = [1.0, 0.0, 0.0, 0.0]
    return struct.pack(f"{EMBEDDING_DIM}f", *vals)


def _make_conn(db_path: str) -> sqlite3.Connection:
    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def run_test(db_path: str) -> None:
    conn = _make_conn(db_path)
    init_schema(conn, EMBEDDING_DIM)

    # Verify the four new columns exist after init
    cols = {row[1] for row in conn.execute("PRAGMA table_info(python_symbols)").fetchall()}
    for col in ("parent_impl", "visibility", "is_signature_only", "node_type"):
        assert col in cols, f"Column {col!r} missing from python_symbols after init_schema"
    print("PASS: all four columns present after init_schema")

    # Insert a row with all four new columns populated
    import json, uuid
    from datetime import datetime

    sym_id = "pysym-test-" + uuid.uuid4().hex[:8]
    now = datetime.now().isoformat()
    embedding = fake_embed("test")

    # Also ensure the lazy-migration columns exist (content_hash, symbol_id are
    # added by _ensure_python_symbol_hash_columns, not SCHEMA_SQL).
    ps_cols = {row[1] for row in conn.execute("PRAGMA table_info(python_symbols)").fetchall()}
    if "content_hash" not in ps_cols:
        conn.execute("ALTER TABLE python_symbols ADD COLUMN content_hash TEXT")
    if "symbol_id" not in ps_cols:
        conn.execute("ALTER TABLE python_symbols ADD COLUMN symbol_id TEXT")
    conn.commit()

    conn.execute("""
        INSERT INTO python_symbols
            (id, name, kind, module, signature, status, is_lru_cached,
             frame_hint, redirect_to, docstring_summary, lean_citations,
             kb_refs, also_in_modules, file, line, project, created_at, updated_at,
             embedding, content_hash, symbol_id,
             parent_impl, visibility, is_signature_only, node_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?)
    """, (sym_id, "greet", "function", "mylib::hello", "pub fn greet(name: &str) -> String",
          "public", 0, None, None, "Says hello", "[]", "[]", "[]",
          "/src/hello.rs", 5, "test-project", now, now,
          embedding, "fakehash", sym_id,
          "Greet for Greeter", "pub", 0, "function_item"))
    conn.commit()

    row = conn.execute(
        "SELECT parent_impl, visibility, is_signature_only, node_type "
        "FROM python_symbols WHERE id = ?", (sym_id,)
    ).fetchone()

    assert row is not None, "Row not found after insert"
    assert row["parent_impl"] == "Greet for Greeter", f"parent_impl mismatch: {row['parent_impl']!r}"
    assert row["visibility"] == "pub", f"visibility mismatch: {row['visibility']!r}"
    assert row["is_signature_only"] == 0, f"is_signature_only mismatch: {row['is_signature_only']!r}"
    assert row["node_type"] == "function_item", f"node_type mismatch: {row['node_type']!r}"
    print("PASS: all four columns round-trip correctly after direct INSERT")

    conn.close()


def test_via_facade(db_path: str) -> None:
    """Test round-trip through KnowledgeBase.add_python_symbol (no real embedding server)."""
    import importlib
    import unittest.mock as mock

    # Patch EmbeddingService.embed to avoid network call
    with mock.patch("kb.core.embedding.EmbeddingService.embed", side_effect=fake_embed), \
         mock.patch("kb.facade.KnowledgeBase._embed", side_effect=fake_embed):

        from kb.facade import KnowledgeBase
        kb = KnowledgeBase(db_path=Path(db_path), embedding_dim=EMBEDDING_DIM)

        result = kb.add_python_symbol(
            name="render",
            kind="function",
            module="mylib::render",
            signature="pub fn render(&self) -> String",
            file="/src/render.rs",
            line=10,
            status="public",
            project="test-project",
            parent_impl="Renderer",
            visibility="pub",
            is_signature_only=False,
            node_type="function_item",
        )
        assert result["is_new"] is True, f"Expected is_new=True, got {result}"
        assert result["parent_impl"] == "Renderer", f"parent_impl in return: {result}"
        assert result["visibility"] == "pub", f"visibility in return: {result}"
        assert result["is_signature_only"] is False, f"is_signature_only in return: {result}"
        assert result["node_type"] == "function_item", f"node_type in return: {result}"
        print("PASS: add_python_symbol returns correct metadata")

        # Read back from DB
        conn = kb.conn
        row = conn.execute(
            "SELECT parent_impl, visibility, is_signature_only, node_type "
            "FROM python_symbols WHERE name=? AND module=?",
            ("render", "mylib::render")
        ).fetchone()
        assert row is not None, "Row not found after add_python_symbol"
        assert row["parent_impl"] == "Renderer"
        assert row["visibility"] == "pub"
        assert row["is_signature_only"] == 0
        assert row["node_type"] == "function_item"
        print("PASS: DB row has correct metadata after add_python_symbol")

        # Also test a signature-only chunk
        result2 = kb.add_python_symbol(
            name="sig_only",
            kind="function",
            module="mylib::sig",
            signature="pub fn sig_only(x: i32)",
            file="/src/sig.rs",
            line=20,
            status="public",
            project="test-project",
            parent_impl=None,
            visibility="pub(crate)",
            is_signature_only=True,
            node_type="function_item",
        )
        row2 = conn.execute(
            "SELECT parent_impl, visibility, is_signature_only, node_type "
            "FROM python_symbols WHERE name=? AND module=?",
            ("sig_only", "mylib::sig")
        ).fetchone()
        assert row2["is_signature_only"] == 1, f"Expected 1 got {row2['is_signature_only']}"
        assert row2["parent_impl"] is None
        assert row2["visibility"] == "pub(crate)"
        print("PASS: signature-only chunk round-trips is_signature_only=1 and NULL parent_impl")

        kb.close()


def test_migration_idempotent(db_path: str) -> None:
    """Run init_schema twice on the same DB; second call must not error."""
    conn = _make_conn(db_path)
    init_schema(conn, EMBEDDING_DIM)
    init_schema(conn, EMBEDDING_DIM)  # second run — must be idempotent
    print("PASS: init_schema is idempotent (ran twice without error)")
    conn.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db1 = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db2 = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db3 = f.name

    try:
        print("=== Test 1: direct INSERT round-trip ===")
        run_test(db1)

        print("\n=== Test 2: facade add_python_symbol round-trip ===")
        test_via_facade(db2)

        print("\n=== Test 3: migration idempotency ===")
        test_migration_idempotent(db3)

        print("\nAll tests passed.")
    finally:
        for p in (db1, db2, db3):
            try:
                os.unlink(p)
            except OSError:
                pass
