"""Test the python_symbols->symbols migration on a throwaway DB (dim=8)."""
import os, sqlite3, tempfile
import sqlite_vec
from kb.core.schema import init_schema

db = os.path.join(tempfile.mkdtemp(), "old.db")
conn = sqlite3.connect(db)
conn.enable_load_extension(True)
sqlite_vec.load(conn)

# Seed the OLD schema with one symbol + its embedding.
# Mirror the CURRENT live column set (project + chunker cols already present via past ALTERs).
conn.execute("CREATE TABLE python_symbols (id TEXT PRIMARY KEY, name TEXT, kind TEXT, "
             "module TEXT, signature TEXT, status TEXT, file TEXT, line INTEGER, "
             "project TEXT, parent_impl TEXT, visibility TEXT, is_signature_only INTEGER, node_type TEXT)")
conn.execute("CREATE VIRTUAL TABLE python_symbols_vec USING vec0(id TEXT PRIMARY KEY, embedding float[8])")
conn.execute("INSERT INTO python_symbols(id,name,kind,module,signature,status,file,line) "
             "VALUES('s1','foo','function','m','sig','public','f.py',1)")
conn.execute("INSERT INTO python_symbols_vec(id, embedding) VALUES('s1', ?)",
             [sqlite_vec.serialize_float32([0.1] * 8)])
conn.commit()

init_schema(conn, 8)

tbls = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert "symbols" in tbls, "symbols table missing"
assert "python_symbols" not in tbls, "legacy python_symbols still present"
assert "python_symbols_vec" not in tbls, "legacy vec still present"
assert conn.execute("SELECT name FROM symbols WHERE id='s1'").fetchone()[0] == "foo", "row lost"
assert conn.execute("SELECT id FROM symbols_vec WHERE id='s1'").fetchone() is not None, "embedding lost"
# new columns added by the ALTER blocks
cols = {r[1] for r in conn.execute("PRAGMA table_info(symbols)")}
assert {"project", "parent_impl", "node_type"} <= cols, f"missing cols: {cols}"
print("MIGRATION OK — data + embedding preserved, legacy dropped, new cols present")

# idempotency: second run must not error or lose data
init_schema(conn, 8)
assert conn.execute("SELECT name FROM symbols WHERE id='s1'").fetchone()[0] == "foo"
print("IDEMPOTENT OK")
