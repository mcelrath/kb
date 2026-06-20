"""Tests for document_sections schema + DocumentSectionsRepository (Phase 1).

Covers:
1. Fresh DB: init_schema creates document_sections and document_sections_vec.
2. DocumentsRepository.add() persists source_path / source_hash.
3. DocumentSectionsRepository.add() / get() / list_by_document().
4. breadcrumb() walks parent chain root-to-leaf.
5. supersede() marks old row and sets superseded_by.
6. upsert_by_path(): no-op on same hash; supersedes on changed hash.
7. KNN round-trip: insert embedding into document_sections_vec, SELECT … MATCH returns the row.
"""

import sqlite3

import pytest

from kb.core.schema import init_schema
from kb.entities.document_sections import DocumentSectionsRepository
from kb.entities.documents import DocumentsRepository

EMBEDDING_DIM = 4  # tiny dim for tests


@pytest.fixture()
def conn():
    """In-memory SQLite with full schema (including vec0 virtual tables)."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    # vec0 requires sqlite-vec extension; load it the same way the main app does.
    try:
        import sqlite_vec
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
    except Exception:
        # If the extension is missing the vec0 tests will raise; that's a CI setup issue,
        # not a schema issue. Other tests (non-vec0) still run.
        pass
    init_schema(c, EMBEDDING_DIM)
    return c


@pytest.fixture()
def docs(conn):
    return DocumentsRepository(conn)


@pytest.fixture()
def sections(conn):
    return DocumentSectionsRepository(conn)


@pytest.fixture()
def doc_id(docs):
    return docs.add(
        title="Test Doc",
        doc_type="reference",
        source_path="/tmp/test.pdf",
        source_hash="abc123",
    )


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------

def test_document_sections_table_exists(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "document_sections" in tables


def test_document_sections_vec_table_exists(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','shadow','virtual')"
    )}
    # vec0 shows up in sqlite_master as a table (virtual)
    assert "document_sections_vec" in {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master")
    }


# ---------------------------------------------------------------------------
# 2. documents.source_path / source_hash
# ---------------------------------------------------------------------------

def test_documents_add_source_fields(conn, doc_id):
    row = conn.execute(
        "SELECT source_path, source_hash FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    assert row["source_path"] == "/tmp/test.pdf"
    assert row["source_hash"] == "abc123"


def test_documents_add_source_fields_optional(docs):
    doc_id = docs.add(title="No source", doc_type="internal")
    row = docs.get(doc_id)
    assert row is not None
    # source_* not in get() return dict but the column exists; check via SQL
    # (get() returns a subset — that's fine; the insert must not fail)


# ---------------------------------------------------------------------------
# 3. Basic CRUD
# ---------------------------------------------------------------------------

def test_add_and_get_section(sections, doc_id):
    sid = sections.add(
        document_id=doc_id,
        path="1",
        level=1,
        ordinal=0,
        heading="Introduction",
        content="Hello world.",
        kind="prose",
        content_hash="h1",
    )
    assert sid.startswith("sec-")
    row = sections.get(sid)
    assert row is not None
    assert row["heading"] == "Introduction"
    assert row["document_id"] == doc_id
    assert row["status"] == "active"


def test_list_by_document_ordered_by_ordinal(sections, doc_id):
    ids = []
    for i in range(3):
        ids.append(sections.add(
            document_id=doc_id,
            path=str(i),
            level=1,
            ordinal=i,
            kind="prose",
            content_hash=f"h{i}",
        ))
    rows = sections.list_by_document(doc_id)
    assert [r["id"] for r in rows] == ids


def test_invalid_kind_raises(sections, doc_id):
    with pytest.raises(ValueError, match="Invalid kind"):
        sections.add(
            document_id=doc_id,
            path="x",
            level=1,
            ordinal=0,
            kind="video",
            content_hash="hx",
        )


# ---------------------------------------------------------------------------
# 4. breadcrumb()
# ---------------------------------------------------------------------------

def test_breadcrumb_root_to_leaf(sections, doc_id):
    root_id = sections.add(
        document_id=doc_id, path="1", level=1, ordinal=0,
        heading="Chapter 1", kind="prose", content_hash="r",
    )
    child_id = sections.add(
        document_id=doc_id, path="1.1", level=2, ordinal=1,
        heading="Section 1.1", kind="prose", content_hash="c",
        parent_section_id=root_id,
    )
    leaf_id = sections.add(
        document_id=doc_id, path="1.1.1", level=3, ordinal=2,
        heading="Sub-section 1.1.1", kind="prose", content_hash="l",
        parent_section_id=child_id,
    )

    crumb = sections.breadcrumb(leaf_id)
    assert len(crumb) == 3
    assert crumb[0]["id"] == root_id
    assert crumb[1]["id"] == child_id
    assert crumb[2]["id"] == leaf_id
    assert crumb[0]["heading"] == "Chapter 1"
    assert crumb[2]["path"] == "1.1.1"


def test_breadcrumb_single_section(sections, doc_id):
    sid = sections.add(
        document_id=doc_id, path="1", level=1, ordinal=0,
        kind="prose", content_hash="s",
    )
    crumb = sections.breadcrumb(sid)
    assert len(crumb) == 1
    assert crumb[0]["id"] == sid


def test_breadcrumb_nonexistent_returns_empty(sections):
    assert sections.breadcrumb("sec-does-not-exist") == []


# ---------------------------------------------------------------------------
# 5. supersede()
# ---------------------------------------------------------------------------

def test_supersede(sections, doc_id):
    old_id = sections.add(
        document_id=doc_id, path="1", level=1, ordinal=0,
        kind="prose", content_hash="old",
    )
    new_id = sections.add(
        document_id=doc_id, path="1", level=1, ordinal=0,
        kind="prose", content_hash="new",
    )
    result = sections.supersede(old_id, new_id)
    assert result is True

    old_row = sections.get(old_id)
    assert old_row["status"] == "superseded"
    assert old_row["superseded_by"] == new_id


# ---------------------------------------------------------------------------
# 6. upsert_by_path()
# ---------------------------------------------------------------------------

def test_upsert_noop_same_hash(sections, doc_id):
    sid, created = sections.upsert_by_path(
        document_id=doc_id, path="u1", content_hash="same",
        level=1, ordinal=0, kind="prose",
    )
    assert created is True

    sid2, created2 = sections.upsert_by_path(
        document_id=doc_id, path="u1", content_hash="same",
        level=1, ordinal=0, kind="prose",
    )
    assert created2 is False
    assert sid2 == sid


def test_upsert_supersedes_on_changed_hash(sections, doc_id):
    sid, _ = sections.upsert_by_path(
        document_id=doc_id, path="u2", content_hash="v1",
        level=1, ordinal=0, kind="prose",
    )

    sid2, created2 = sections.upsert_by_path(
        document_id=doc_id, path="u2", content_hash="v2",
        level=1, ordinal=0, kind="prose",
    )
    assert created2 is True
    assert sid2 != sid

    old = sections.get(sid)
    assert old["status"] == "superseded"
    assert old["superseded_by"] == sid2


# ---------------------------------------------------------------------------
# 7. vec0 KNN round-trip
# ---------------------------------------------------------------------------

def test_vec_round_trip(conn, sections, doc_id):
    """Insert embedding into document_sections_vec; SELECT … MATCH returns it."""
    import struct

    # Skip if vec0 not available
    try:
        conn.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        pytest.skip("sqlite-vec extension not available")

    sid = sections.add(
        document_id=doc_id, path="v1", level=1, ordinal=0,
        kind="prose", content_hash="vc",
        embed_text="hello world",
    )

    # Serialize embedding as little-endian float32 blob
    query_vec = [1.0, 0.0, 0.0, 0.0]
    blob = struct.pack(f"{EMBEDDING_DIM}f", *query_vec)

    conn.execute(
        "INSERT INTO document_sections_vec(id, embedding) VALUES (?, ?)",
        (sid, blob),
    )
    conn.commit()

    rows = conn.execute(
        """SELECT ds.id, ds.heading, dsv.distance
           FROM document_sections ds
           JOIN document_sections_vec dsv ON ds.id = dsv.id
           WHERE dsv.embedding MATCH ?
             AND k = 1""",
        (blob,),
    ).fetchall()

    assert len(rows) == 1
    assert rows[0]["id"] == sid
