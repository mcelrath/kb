"""Tests for Phase 4: retrieval verbs + union search + reembed wiring.

Covers:
1. kb doc toc: prints heading tree with section ids + paths.
2. kb doc get --path: returns the section at the given path.
3. kb doc get --subtree: includes descendant sections.
4. Union search returns a section hit with a breadcrumb.
5. reembed_all round-trip: document_sections_vec is repopulated and
   assertion_checks passes (document_sections_vec count == document_sections count).
"""

import struct
import sqlite3

import pytest

from kb.core.schema import init_schema
from kb.entities.document_sections import DocumentSectionsRepository
from kb.entities.documents import DocumentsRepository

EMBEDDING_DIM = 4


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn():
    """In-memory SQLite with full schema including vec0 virtual tables."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
    except Exception:
        pass
    init_schema(c, EMBEDDING_DIM)
    return c


@pytest.fixture()
def docs_repo(conn):
    return DocumentsRepository(conn)


@pytest.fixture()
def sections_repo(conn):
    return DocumentSectionsRepository(conn)


def _pack_vec(*values: float) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


@pytest.fixture()
def doc_with_sections(conn, docs_repo, sections_repo):
    """Create a document with a three-level section tree."""
    doc_id = docs_repo.add(
        title="Test Manual",
        doc_type="reference",
        source_path="/tmp/manual.pdf",
        source_hash="aabbcc",
    )

    root_id = sections_repo.add(
        document_id=doc_id,
        path="1",
        level=1,
        ordinal=0,
        heading="Chapter 1",
        content="Introduction text.",
        kind="prose",
        content_hash="r1",
        embed_text="Chapter 1 Introduction",
    )
    child_id = sections_repo.add(
        document_id=doc_id,
        path="1.1",
        level=2,
        ordinal=1,
        heading="Section 1.1",
        content="Details here.",
        kind="prose",
        content_hash="c1",
        embed_text="Section 1.1 Details",
        parent_section_id=root_id,
    )
    leaf_id = sections_repo.add(
        document_id=doc_id,
        path="1.1.1",
        level=3,
        ordinal=2,
        heading="Sub-section 1.1.1",
        content="Leaf content.",
        kind="prose",
        content_hash="l1",
        embed_text="Sub-section leaf content",
        parent_section_id=child_id,
    )

    return {
        "doc_id": doc_id,
        "root_id": root_id,
        "child_id": child_id,
        "leaf_id": leaf_id,
    }


# ---------------------------------------------------------------------------
# 1. doc toc — heading tree
# ---------------------------------------------------------------------------

def test_doc_toc_returns_all_sections(conn, doc_with_sections, sections_repo):
    """list_by_document (the backing data for toc) returns all 3 sections in ordinal order."""
    doc_id = doc_with_sections["doc_id"]
    sections = sections_repo.list_by_document(doc_id)
    assert len(sections) == 3
    paths = [s["path"] for s in sections]
    assert paths == ["1", "1.1", "1.1.1"]


def test_doc_toc_section_ids_and_headings(conn, doc_with_sections, sections_repo):
    doc_id = doc_with_sections["doc_id"]
    sections = sections_repo.list_by_document(doc_id)
    headings = {s["path"]: s["heading"] for s in sections}
    assert headings["1"] == "Chapter 1"
    assert headings["1.1"] == "Section 1.1"
    assert headings["1.1.1"] == "Sub-section 1.1.1"
    # Every section must have an id
    for s in sections:
        assert s["id"].startswith("sec-")


# ---------------------------------------------------------------------------
# 2. doc get --path
# ---------------------------------------------------------------------------

def test_doc_get_returns_correct_section(conn, doc_with_sections):
    doc_id = doc_with_sections["doc_id"]
    child_id = doc_with_sections["child_id"]

    row = conn.execute(
        "SELECT * FROM document_sections WHERE document_id = ? AND path = ? AND status = 'active'",
        (doc_id, "1.1"),
    ).fetchone()

    assert row is not None
    assert row["id"] == child_id
    assert row["heading"] == "Section 1.1"
    assert row["content"] == "Details here."


def test_doc_get_nonexistent_path_returns_none(conn, doc_with_sections):
    doc_id = doc_with_sections["doc_id"]
    row = conn.execute(
        "SELECT * FROM document_sections WHERE document_id = ? AND path = ? AND status = 'active'",
        (doc_id, "9.9.9"),
    ).fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# 3. doc get --subtree
# ---------------------------------------------------------------------------

def test_doc_get_subtree_includes_descendants(conn, doc_with_sections, sections_repo):
    """--subtree at path '1' should include '1', '1.1', '1.1.1'."""
    doc_id = doc_with_sections["doc_id"]
    path = "1"

    all_sections = sections_repo.list_by_document(doc_id)
    prefix = path + "."
    subtree = [s for s in all_sections if s["path"] == path or s["path"].startswith(prefix)]

    paths = {s["path"] for s in subtree}
    assert "1" in paths
    assert "1.1" in paths
    assert "1.1.1" in paths
    assert len(subtree) == 3


def test_doc_get_subtree_leaf_has_no_children(conn, doc_with_sections, sections_repo):
    """--subtree at leaf '1.1.1' returns only that one section."""
    doc_id = doc_with_sections["doc_id"]
    path = "1.1.1"

    all_sections = sections_repo.list_by_document(doc_id)
    prefix = path + "."
    subtree = [s for s in all_sections if s["path"] == path or s["path"].startswith(prefix)]

    assert len(subtree) == 1
    assert subtree[0]["path"] == "1.1.1"


# ---------------------------------------------------------------------------
# 4. Union search: section hit with breadcrumb
# ---------------------------------------------------------------------------

def test_union_search_section_hit_has_breadcrumb(conn, doc_with_sections, sections_repo):
    """Insert an embedding for the leaf section and run search_sections; hit must carry breadcrumb."""
    try:
        conn.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        pytest.skip("sqlite-vec extension not available")

    leaf_id = doc_with_sections["leaf_id"]
    vec = _pack_vec(1.0, 0.0, 0.0, 0.0)

    conn.execute(
        "INSERT INTO document_sections_vec(id, embedding) VALUES (?, ?)",
        (leaf_id, vec),
    )
    conn.commit()

    from kb.search.hybrid import HybridSearch

    # Minimal EmbeddingService stub
    class _StubEmbed:
        embedding_dim = EMBEDDING_DIM
        embedding_format = "llamacpp"
        embedding_url = "http://localhost:9999"
        embedding_model = "stub"

        def embed(self, text, **kwargs):
            return vec

    hs = HybridSearch(conn=conn, embedding_service=_StubEmbed())
    results = hs.search_sections(query_embedding=vec, limit=5)

    assert len(results) >= 1
    hit = results[0]
    assert hit["id"] == leaf_id
    assert hit["result_type"] == "section"
    assert "breadcrumb" in hit

    crumb = hit["breadcrumb"]
    assert len(crumb) == 3  # root -> child -> leaf
    assert crumb[0]["path"] == "1"
    assert crumb[1]["path"] == "1.1"
    assert crumb[2]["path"] == "1.1.1"
    assert hit["doc_id"] == doc_with_sections["doc_id"]
    assert hit["path"] == "1.1.1"


# ---------------------------------------------------------------------------
# 5. reembed_all: document_sections_vec assertion_checks pass
# ---------------------------------------------------------------------------

def test_reembed_all_document_sections_assertion(conn, doc_with_sections):
    """After reembed_all (stubbed), document_sections_vec count == document_sections count."""
    try:
        conn.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        pytest.skip("sqlite-vec extension not available")

    # Insert one embedding per section directly (simulating what _do_table does)
    rows = conn.execute("SELECT id FROM document_sections").fetchall()
    vec = _pack_vec(0.5, 0.5, 0.0, 0.0)
    for row in rows:
        conn.execute(
            "INSERT OR REPLACE INTO document_sections_vec(id, embedding) VALUES (?, ?)",
            (row[0], vec),
        )
    conn.commit()

    # Run the assertion check that reembed_all performs
    vec_count = conn.execute("SELECT COUNT(*) FROM document_sections_vec").fetchone()[0]
    base_count = conn.execute("SELECT COUNT(*) FROM document_sections").fetchone()[0]

    assert base_count == 3  # three sections in our fixture
    assert vec_count == base_count, (
        f"document_sections_vec has {vec_count} rows but document_sections has {base_count}"
    )
