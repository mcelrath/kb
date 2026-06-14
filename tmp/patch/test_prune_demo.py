"""Demonstrate PRUNE and content_hash on a temp sqlite DB (no embedding server needed)."""
import sqlite3
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# We test facade logic directly without the embedding server by monkeypatching _embed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kb import KnowledgeBase, DEFAULT_DB_PATH

TMPDB = Path(__file__).parent / "demo.db"
TMPDB.unlink(missing_ok=True)

# Monkeypatch embedding to avoid needing the remote server
def _fake_embed(self, text: str) -> bytes:
    # Return 4096 floats as little-endian f32 bytes (all zeros)
    import struct
    return struct.pack('<4096f', *([0.0] * 4096))

import kb.core.embedding as emb_mod
_orig_embed = emb_mod.EmbeddingService.embed
emb_mod.EmbeddingService.embed = _fake_embed

# Also patch python_symbols_vec insert to use dummy vec table
# (sqlite-vec may not be loaded in test; we guard gracefully)
kb_instance = KnowledgeBase(db_path=TMPDB)

# Ensure columns exist
kb_instance._ensure_python_symbol_hash_columns()

# Helper to insert a symbol without embedding (directly, bypass vec table)
def insert_sym(kb, name, module, sig, doc, fpath, project="test"):
    """Insert via add_python_symbol but silently skip vec errors."""
    try:
        result = kb.add_python_symbol(
            name=name, kind="function", module=module, signature=sig,
            file=str(fpath), line=1, status="public", docstring_summary=doc,
            project=project,
        )
        return result
    except Exception as e:
        # sqlite-vec not loaded in test env — insert manually
        from datetime import datetime
        import json
        content_hash = kb._python_symbol_content_hash(project, module, name, sig, doc)
        symbol_id = kb._python_symbol_stable_id(project, module, name)
        now = datetime.now().isoformat()
        existing = kb.conn.execute(
            "SELECT id, content_hash FROM python_symbols WHERE name=? AND module=?",
            (name, module)
        ).fetchone()
        if existing:
            if existing["content_hash"] == content_hash:
                return {"id": existing["id"], "is_new": False, "skipped": True}
            kb.conn.execute("""
                UPDATE python_symbols SET signature=?, docstring_summary=?,
                    content_hash=?, symbol_id=?, updated_at=?, file=?
                WHERE id=?
            """, (sig, doc, content_hash, symbol_id, now, str(fpath), existing["id"]))
            kb.conn.commit()
            return {"id": existing["id"], "is_new": False}
        kb.conn.execute("""
            INSERT INTO python_symbols
                (id, name, kind, module, signature, status, is_lru_cached,
                 frame_hint, redirect_to, docstring_summary, lean_citations,
                 kb_refs, also_in_modules, file, line, project, created_at, updated_at,
                 content_hash, symbol_id)
            VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, '[]', '[]', '[]', ?, 1, ?, ?, ?, ?, ?)
        """, (symbol_id, name, "function", module, sig, "public", doc,
              str(fpath), project, now, now, content_hash, symbol_id))
        kb.conn.commit()
        return {"id": symbol_id, "is_new": True}

TEST_FILE = Path("/fake/mymodule.py")
MODULE = "mymodule"

# Step 1: Ingest foo + bar
print("=== STEP 1: Ingest foo + bar ===")
r1 = insert_sym(kb_instance, "foo", MODULE, "def foo():", "foo does something", TEST_FILE)
r2 = insert_sym(kb_instance, "bar", MODULE, "def bar(x):", "bar does something else", TEST_FILE)
print(f"  foo: {r1}")
print(f"  bar: {r2}")

rows_before = kb_instance.conn.execute(
    "SELECT name, module, content_hash, symbol_id FROM python_symbols WHERE file=? ORDER BY name",
    (str(TEST_FILE),)
).fetchall()
print("Before prune:")
for r in rows_before:
    print(f"  name={r['name']} module={r['module']} symbol_id={r['symbol_id'][:16]}... content_hash={r['content_hash'][:16]}...")

# Step 2: Re-ingest with bar removed (only foo survives)
print("\n=== STEP 2: Re-ingest (bar removed) — prune stale ===")
live_set = {("foo", MODULE)}
pruned = kb_instance.prune_python_symbols_for_file(str(TEST_FILE), live_set)
print(f"  pruned count: {pruned}")

rows_after = kb_instance.conn.execute(
    "SELECT name, module, content_hash, symbol_id FROM python_symbols WHERE file=? ORDER BY name",
    (str(TEST_FILE),)
).fetchall()
print("After prune:")
for r in rows_after:
    print(f"  name={r['name']} module={r['module']} symbol_id={r['symbol_id'][:16]}... content_hash={r['content_hash'][:16]}...")

# Verify: foo remains, bar is gone
names_after = {r['name'] for r in rows_after}
assert "foo" in names_after, "FAIL: foo should remain"
assert "bar" not in names_after, "FAIL: bar should be pruned"
print("\nPASS: foo remains, bar is pruned")

# Step 3: Verify content_hash populated
print("\n=== STEP 3: Verify content_hash + symbol_id ===")
row = kb_instance.conn.execute(
    "SELECT name, content_hash, symbol_id FROM python_symbols WHERE name='foo'"
).fetchone()
print(f"  foo.content_hash = {row['content_hash']}")
print(f"  foo.symbol_id    = {row['symbol_id']}")
expected_ch = KnowledgeBase._python_symbol_content_hash("test", MODULE, "foo", "def foo():", "foo does something")
expected_id = KnowledgeBase._python_symbol_stable_id("test", MODULE, "foo")
assert row['content_hash'] == expected_ch, f"content_hash mismatch: {row['content_hash']} != {expected_ch}"
assert row['symbol_id'] == expected_id, f"symbol_id mismatch: {row['symbol_id']} != {expected_id}"
print("  PASS: hashes match expected values")

# Step 4: Verify empty-guard (parse failure => no prune)
print("\n=== STEP 4: Verify empty-guard (parse failure => no prune) ===")
insert_sym(kb_instance, "baz", MODULE, "def baz():", "baz", TEST_FILE)
count_before = kb_instance.conn.execute(
    "SELECT COUNT(*) FROM python_symbols WHERE file=?", (str(TEST_FILE),)
).fetchone()[0]
pruned_empty = kb_instance.prune_python_symbols_for_file(str(TEST_FILE), set())
count_after = kb_instance.conn.execute(
    "SELECT COUNT(*) FROM python_symbols WHERE file=?", (str(TEST_FILE),)
).fetchone()[0]
assert pruned_empty == 0, f"FAIL: empty set should prune 0, got {pruned_empty}"
assert count_before == count_after, "FAIL: rows should not change on empty live set"
print(f"  PASS: empty live_set pruned 0 rows (rows unchanged: {count_before})")

TMPDB.unlink()
print("\nAll checks passed.")
