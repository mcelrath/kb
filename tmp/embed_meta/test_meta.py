#!/usr/bin/env python3
"""
Tests for Phase 2: embedding_meta table + embed-status + reembed_all 7-table coverage.

Uses a TEMP db and a monkeypatched EmbeddingService that returns deterministic
vectors of a given dim, so NO live embedding server is needed.
"""
import sys
import os
import json
import tempfile
import sqlite3
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Monkeypatch EmbeddingService BEFORE importing KnowledgeBase
from kb.core import embedding as _emb_mod
from kb.validation import serialize_f32, l2_normalize


class FakeEmbeddingService:
    """Returns deterministic L2-normalized vectors of self.embedding_dim."""
    def __init__(self, url, dim, cache_max=500, embedding_format="llamacpp",
                 embedding_model="", embedding_key=""):
        self.embedding_url = url
        self.embedding_dim = dim
        self.embedding_format = embedding_format
        self.embedding_model = embedding_model
        self.embedding_key = embedding_key
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_max = cache_max

    def _make_vector(self, text: str, dim: int) -> list[float]:
        """Deterministic: sum of char codes mod dim."""
        seed = sum(ord(c) for c in text) % max(dim, 1)
        vec = [float((seed + i) % 100 + 1) for i in range(dim)]
        total = sum(v * v for v in vec) ** 0.5
        return [v / total for v in vec]

    def embed(self, text: str, max_retries=None, timeout=None) -> bytes:
        vec = self._make_vector(text, self.embedding_dim)
        return serialize_f32(vec)

    def embed_raw(self, text: str) -> list[float]:
        return self._make_vector(text, self.embedding_dim)

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed(t) for t in texts]


def make_kb(db_path: str, dim: int = 768, fmt: str = "llamacpp", model: str = "", url: str = "http://fake:9999/embedding"):
    """Create a KnowledgeBase with FakeEmbeddingService injected."""
    os.environ["KB_EMBEDDING_DIM"] = str(dim)
    os.environ["KB_EMBEDDING_FORMAT"] = fmt
    os.environ["KB_EMBEDDING_MODEL"] = model
    os.environ["KB_EMBEDDING_URL"] = url

    # We need to patch BEFORE KnowledgeBase.__init__ runs
    from kb.facade import KnowledgeBase
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.db_path = Path(db_path)
    kb.db_path.parent.mkdir(parents=True, exist_ok=True)
    kb.embedding_url = url
    kb.embedding_dim = dim

    from kb.core.connection import DatabaseConnection
    from kb.core.schema import init_schema
    db_conn = DatabaseConnection(db_path, dim)
    kb.conn = db_conn.conn
    init_schema(kb.conn, dim)

    # Inject fake embedding
    fake_emb = FakeEmbeddingService(url, dim, embedding_format=fmt, embedding_model=model)
    kb._embedding = fake_emb

    # Stub out LLM / analyzer / search / repos to avoid network calls
    from unittest.mock import MagicMock
    kb._llm = MagicMock()
    kb._llm.expand_query = lambda q, p, u, v: q
    kb._analyzer = MagicMock()
    kb._analyzer.generate_summary = lambda c, e=None: None
    kb._analyzer.suggest_tags = lambda c, existing=None: []
    kb._analyzer.classify_type = lambda c: "discovery"
    kb._analyzer.validate_finding = lambda c, t=None: {}

    from kb.search.hybrid import HybridSearch
    kb._search = HybridSearch(kb.conn, fake_emb, expand_query=lambda q, p, v: q)

    from kb.entities.scripts import ScriptsRepository
    from kb.entities.documents import DocumentsRepository
    from kb.entities.theorems import TheoremRepository
    from kb.entities.concepts import ConceptRepository
    from kb.entities.issues import IssuesRepository
    kb._scripts = ScriptsRepository(kb.conn, fake_emb, finding_exists=lambda fid: False)
    kb._documents = DocumentsRepository(kb.conn)
    kb._theorems = TheoremRepository(kb.conn, fake_emb)
    kb._concepts = ConceptRepository(kb.conn, fake_emb)
    kb._issues = IssuesRepository(kb.conn, fake_emb)

    # Seed embedding meta (first-run behavior)
    kb._ensure_embedding_meta()

    return kb


def seed_base_tables(kb, count: int = 3) -> dict:
    """Insert rows in all 7 base tables for testing. Returns inserted IDs."""
    from datetime import datetime
    import uuid
    now = datetime.now().isoformat()
    ids = {t: [] for t in ["findings", "scripts", "lean_theorems", "concepts",
                            "issues", "python_symbols", "tex_annotations"]}

    # findings + findings_vec
    for i in range(count):
        fid = f"kb-test-finding-{i:04d}"
        emb = kb._embedding.embed(f"test finding content {i}")
        kb.conn.execute(
            "INSERT OR IGNORE INTO findings (id, type, status, content, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (fid, "discovery", "current", f"test finding content {i}", now, now)
        )
        kb.conn.execute("DELETE FROM findings_vec WHERE id = ?", (fid,))
        kb.conn.execute("INSERT INTO findings_vec (id, embedding) VALUES (?, ?)", (fid, emb))
        ids["findings"].append(fid)

    # scripts + scripts_vec
    for i in range(count):
        sid = f"script-test-{i:04d}"
        emb = kb._embedding.embed(f"test purpose {i}")
        kb.conn.execute(
            "INSERT OR IGNORE INTO scripts (id, path, filename, content_hash, purpose, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (sid, f"/tmp/s{i}.py", f"s{i}.py", f"hash{i}", f"test purpose {i}", now, now)
        )
        kb.conn.execute("DELETE FROM scripts_vec WHERE id = ?", (sid,))
        kb.conn.execute("INSERT INTO scripts_vec (id, embedding) VALUES (?, ?)", (sid, emb))
        ids["scripts"].append(sid)

    # lean_theorems + lean_theorems_vec
    for i in range(count):
        tid = f"lt-test-{i:04d}"
        emb = kb._embedding.embed(f"theorem statement {i}")
        kb.conn.execute(
            "INSERT OR IGNORE INTO lean_theorems (id, lean_name, name, statement, declaration, file, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (tid, f"Test.Thm{i}", f"Thm{i}", f"theorem statement {i}", f"theorem Thm{i}", f"Test{i}.lean", now, now)
        )
        kb.conn.execute("DELETE FROM lean_theorems_vec WHERE id = ?", (tid,))
        kb.conn.execute("INSERT INTO lean_theorems_vec (id, embedding) VALUES (?, ?)", (tid, emb))
        ids["lean_theorems"].append(tid)

    # concepts + concepts_vec
    for i in range(count):
        cid = f"concept-test-{i:04d}"
        emb = kb._embedding.embed(f"test concept claim {i}")
        kb.conn.execute(
            "INSERT OR IGNORE INTO concepts (id, domain, claim, created_at, updated_at) VALUES (?,?,?,?,?)",
            (cid, "math", f"test concept claim {i}", now, now)
        )
        kb.conn.execute("DELETE FROM concepts_vec WHERE id = ?", (cid,))
        kb.conn.execute("INSERT INTO concepts_vec (id, embedding) VALUES (?, ?)", (cid, emb))
        ids["concepts"].append(cid)

    # issues + issues_vec
    for i in range(count):
        iid = f"issue-test-{i:04d}"
        emb = kb._embedding.embed(f"test issue {i}")
        kb.conn.execute(
            "INSERT OR IGNORE INTO issues (id, type, status, title, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (iid, "task", "open", f"test issue {i}", now, now)
        )
        kb.conn.execute("DELETE FROM issues_vec WHERE id = ?", (iid,))
        kb.conn.execute("INSERT INTO issues_vec (id, embedding) VALUES (?, ?)", (iid, emb))
        ids["issues"].append(iid)

    # python_symbols (with embedding BLOB written inline)
    for i in range(count):
        pid = f"pysym-test-{i:04d}"
        emb_text = f"mymod.func{i}: (arg{i}: int) -> str test docstring {i}"
        emb = kb._embedding.embed(emb_text)
        kb._ensure_python_symbol_hash_columns()
        kb.conn.execute(
            "INSERT OR IGNORE INTO python_symbols "
            "(id, name, kind, module, signature, status, file, line, created_at, updated_at, embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pid, f"func{i}", "function", "mymod", f"(arg{i}: int) -> str", "public",
             f"/mymod.py", i+1, now, now, emb)
        )
        # seed _vec
        kb.conn.execute("DELETE FROM python_symbols_vec WHERE id = ?", (pid,))
        kb.conn.execute("INSERT INTO python_symbols_vec (id, embedding) VALUES (?, ?)", (pid, emb))
        ids["python_symbols"].append(pid)

    # tex_annotations (with embedding BLOB)
    for i in range(count):
        aid = f"texann-test-{i:04d}"
        emb_text = f"Section {i} label{i} python:[] lean:[] context {i}"
        emb = kb._embedding.embed(emb_text)
        kb.conn.execute(
            "INSERT OR IGNORE INTO tex_annotations "
            "(id, file, line, section_label, section_title, context, created_at, updated_at, embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, f"paper{i}.tex", i*10+1, f"label{i}", f"Section {i}", f"context {i}", now, now, emb)
        )
        kb.conn.execute("DELETE FROM tex_annotations_vec WHERE id = ?", (aid,))
        kb.conn.execute("INSERT INTO tex_annotations_vec (id, embedding) VALUES (?, ?)", (aid, emb))
        ids["tex_annotations"].append(aid)

    kb.conn.commit()
    return ids


# =============================================================================
# Test 1: no-meta -> _ensure seeds it -> status ok
# =============================================================================
def test_no_meta_then_seed():
    print("\n--- TEST 1: no-meta seed ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        kb = make_kb(db_path, dim=768)

        # _ensure_embedding_meta was called in make_kb; row should exist
        status = kb.embedding_status()
        print(f"  verdict after seed: {status['verdict']}")
        assert status["verdict"] == "ok", f"Expected ok, got {status['verdict']}"
        assert status["stored"] is not None
        assert status["stored"]["dim"] == 768
        print("  PASS")


# =============================================================================
# Test 2: same-dim model change -> mismatch-same-dim
# =============================================================================
def test_same_dim_model_change():
    print("\n--- TEST 2: same-dim model change ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        kb = make_kb(db_path, dim=768, model="model-a")
        status = kb.embedding_status()
        assert status["verdict"] == "ok"

        # Change model without re-seeding
        kb._embedding.embedding_model = "model-b"
        status2 = kb.embedding_status()
        print(f"  verdict after model change: {status2['verdict']}")
        assert status2["verdict"] == "mismatch-same-dim", \
            f"Expected mismatch-same-dim, got {status2['verdict']}"
        print("  PASS")


# =============================================================================
# Test 3: dim change 768->1024: reembed recreates ALL 7 _vec; rowcounts match
# =============================================================================
def test_dim_change_reembed():
    print("\n--- TEST 3: dim change 768->1024 full round-trip ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Create KB at 768
        kb768 = make_kb(db_path, dim=768, model="model-768")
        seed_base_tables(kb768, count=3)

        # Verify initial vec counts at 768
        for vtable in ["findings_vec", "scripts_vec", "lean_theorems_vec",
                       "concepts_vec", "issues_vec", "python_symbols_vec", "tex_annotations_vec"]:
            cnt = kb768.conn.execute(f"SELECT COUNT(*) FROM {vtable}").fetchone()[0]
            assert cnt == 3, f"Expected 3 in {vtable} at 768-setup, got {cnt}"

        # Now simulate dim change: new KB instance at 1024
        kb768.conn.close()

        kb1024 = make_kb(db_path, dim=1024, model="model-1024")

        # Status should show mismatch-dim-change (stored=768 vs configured=1024)
        status = kb1024.embedding_status()
        print(f"  verdict before reembed: {status['verdict']}")
        print(f"  stored_dim={status['stored']['dim']}, configured_dim={status['configured']['dim']}")
        assert status["verdict"] == "mismatch-dim-change", \
            f"Expected mismatch-dim-change, got {status['verdict']}"

        # Run reembed_all (with force_dim to exercise the dim-change path)
        print("  Running reembed_all...")
        result = kb1024.reembed_all(resume=False)

        # Verify all 7 _vec tables have 3 rows at new dim
        BASE_COUNTS = {
            "findings_vec":        ("SELECT COUNT(*) FROM findings", 3),
            "scripts_vec":         ("SELECT COUNT(*) FROM scripts", 3),
            "lean_theorems_vec":   ("SELECT COUNT(*) FROM lean_theorems", 3),
            "concepts_vec":        ("SELECT COUNT(*) FROM concepts", 3),
            "issues_vec":          ("SELECT COUNT(*) FROM issues", 3),
            "python_symbols_vec":  ("SELECT COUNT(*) FROM python_symbols WHERE embedding IS NOT NULL", 3),
            "tex_annotations_vec": ("SELECT COUNT(*) FROM tex_annotations WHERE embedding IS NOT NULL", 3),
        }
        for vtable, (base_sql, expected) in BASE_COUNTS.items():
            vec_cnt = kb1024.conn.execute(f"SELECT COUNT(*) FROM {vtable}").fetchone()[0]
            base_cnt = kb1024.conn.execute(base_sql).fetchone()[0]
            print(f"  {vtable}: vec={vec_cnt} base={base_cnt} expected={expected}")
            assert vec_cnt == expected, f"{vtable} vec count wrong: {vec_cnt} != {expected}"
            assert base_cnt == expected, f"{base_sql} base count wrong: {base_cnt} != {expected}"

        # Verify embedding_meta updated to 1024
        status2 = kb1024.embedding_status()
        print(f"  verdict after reembed: {status2['verdict']}")
        assert status2["verdict"] == "ok", f"Expected ok after reembed, got {status2['verdict']}"
        assert status2["stored"]["dim"] == 1024, f"stored dim should be 1024: {status2['stored']['dim']}"

        # Verify python_symbols base BLOB was updated to 1024-dim vectors
        rows = kb1024.conn.execute(
            "SELECT id, embedding FROM python_symbols WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            blob = row[1]
            import struct
            actual_dim = len(struct.unpack(f"{len(blob)//4}f", blob))
            assert actual_dim == 1024, f"python_symbols blob dim wrong: {actual_dim} != 1024"

        print("  PASS")


# =============================================================================
# Test 4: py_compile check (done via subprocess)
# =============================================================================
def test_py_compile():
    print("\n--- TEST 4: py_compile check ---")
    import subprocess
    files = [
        str(PROJECT_ROOT / "kb" / "core" / "schema.py"),
        str(PROJECT_ROOT / "kb" / "facade.py"),
        str(PROJECT_ROOT / "kb.py"),
    ]
    result = subprocess.run(
        [sys.executable, "-m", "py_compile"] + files,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr}")
    else:
        print("  PASS: all files compile clean")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    test_py_compile()
    test_no_meta_then_seed()
    test_same_dim_model_change()
    test_dim_change_reembed()
    print("\n=== ALL TESTS PASSED ===")
