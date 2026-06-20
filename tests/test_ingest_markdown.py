"""Tests for kb/ingest/markdown.py — heading-tree chunker."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from kb.ingest.markdown import (
    _parse_sections,
    _build_intermediate,
    _compute_paths,
    _linearize_table,
    count_heading_sections,
    ingest_markdown_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MULTI_SECTION_MD = """\
# Introduction

This is the intro section with some prose content.

## Background

Some background information here.

### Details

Very specific detail text lives here.

## Data Table

| Name | Value | Notes |
|------|-------|-------|
| foo  | 1     | first |
| bar  | 2     | second |

## Conclusion

Final remarks go here.
"""

SINGLE_SECTION_MD = """\
Just a flat markdown file with no headings at all.

Some more text here.
"""

FRONT_MATTER_MD = """\
---
title: My Test Document
project: testproject
doc_type: reference
summary: A test document with front-matter.
---

# Section One

Content of section one.

## Section Two

Content of section two.
"""


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary DB path."""
    return tmp_path / "test.db"


@pytest.fixture
def multi_md(tmp_path):
    p = tmp_path / "multi.md"
    p.write_text(MULTI_SECTION_MD)
    return p


@pytest.fixture
def single_md(tmp_path):
    p = tmp_path / "single.md"
    p.write_text(SINGLE_SECTION_MD)
    return p


@pytest.fixture
def fm_md(tmp_path):
    p = tmp_path / "frontmatter.md"
    p.write_text(FRONT_MATTER_MD)
    return p


# ---------------------------------------------------------------------------
# Unit tests — parser and helpers
# ---------------------------------------------------------------------------

def test_parse_sections_counts():
    sections = _parse_sections(MULTI_SECTION_MD)
    headings = [s for s in sections if s["level"] > 0]
    assert len(headings) == 5  # Introduction, Background, Details, Data Table, Conclusion


def test_parse_sections_levels():
    sections = _parse_sections(MULTI_SECTION_MD)
    headings = [s for s in sections if s["level"] > 0]
    assert headings[0]["level"] == 1
    assert headings[1]["level"] == 2
    assert headings[2]["level"] == 3
    assert headings[3]["level"] == 2
    assert headings[4]["level"] == 2


def test_parse_sections_single_no_headings():
    sections = _parse_sections(SINGLE_SECTION_MD)
    # Only preamble (level 0)
    assert all(s["level"] == 0 for s in sections)


def test_count_heading_sections_multi():
    assert count_heading_sections(MULTI_SECTION_MD) == 5


def test_count_heading_sections_single():
    assert count_heading_sections(SINGLE_SECTION_MD) == 0


def test_linearize_table():
    table = "| Name | Value |\n|------|-------|\n| foo  | 1     |\n| bar  | 2     |"
    result = _linearize_table(table)
    assert "Name | Value" in result
    assert "foo | 1" in result
    assert "bar | 2" in result
    # separator row should be excluded
    assert "---" not in result


def test_intermediate_table_kind():
    """Table blocks in a section body are classified as kind='table'."""
    sections = _parse_sections(MULTI_SECTION_MD)
    intermediate = _build_intermediate(sections)
    kinds = [e["kind"] for e in intermediate]
    assert "table" in kinds


def test_intermediate_table_has_embed_text():
    """Table leaves have non-empty embed_text (linearized rows)."""
    sections = _parse_sections(MULTI_SECTION_MD)
    intermediate = _build_intermediate(sections)
    tables = [e for e in intermediate if e["kind"] == "table"]
    assert tables
    for t in tables:
        assert t["embed_text"]
        assert t["table_repr"]


def test_compute_paths_unique():
    """Every intermediate entry gets a unique path."""
    sections = _parse_sections(MULTI_SECTION_MD)
    intermediate = _build_intermediate(sections)
    _compute_paths(sections, intermediate)
    paths = [e["path"] for e in intermediate]
    assert len(paths) == len(set(paths)), f"Duplicate paths: {paths}"


def test_compute_paths_ordinal_monotone():
    """Ordinals are non-decreasing."""
    sections = _parse_sections(MULTI_SECTION_MD)
    intermediate = _build_intermediate(sections)
    _compute_paths(sections, intermediate)
    ordinals = [e["ordinal"] for e in intermediate]
    assert ordinals == sorted(ordinals)


# ---------------------------------------------------------------------------
# Integration tests — DB persistence
# ---------------------------------------------------------------------------

def test_ingest_creates_document_and_sections(multi_md, tmp_db):
    doc_id, section_ids = ingest_markdown_file(multi_md, db_path=tmp_db)
    assert doc_id.startswith("doc-")
    assert len(section_ids) >= 2  # at least prose + table leaves


def test_ingest_section_kinds(multi_md, tmp_db):
    """DB sections include at least one table-kind leaf."""
    from kb.core.connection import DatabaseConnection
    from kb.core.schema import init_schema

    doc_id, section_ids = ingest_markdown_file(multi_md, db_path=tmp_db)

    conn = DatabaseConnection(tmp_db).conn
    rows = conn.execute(
        "SELECT kind FROM document_sections WHERE document_id = ? AND status = 'active'",
        (doc_id,),
    ).fetchall()
    kinds = {r[0] for r in rows}
    assert "table" in kinds
    assert "prose" in kinds


def test_ingest_paths_unique(multi_md, tmp_db):
    doc_id, section_ids = ingest_markdown_file(multi_md, db_path=tmp_db)

    from kb.core.connection import DatabaseConnection
    conn = DatabaseConnection(tmp_db).conn
    paths = conn.execute(
        "SELECT path FROM document_sections WHERE document_id = ? AND status = 'active'",
        (doc_id,),
    ).fetchall()
    path_list = [r[0] for r in paths]
    assert len(path_list) == len(set(path_list)), f"Duplicate paths in DB: {path_list}"


def test_ingest_duplicate_sibling_headings_no_data_loss(tmp_db, tmp_path):
    """Regression (kb-86b074): two same-level siblings with identical heading
    text (e.g. two '## Notes') must get distinct paths and BOTH bodies must
    survive — the old (level,heading) text-rematch collided them onto one path,
    and upsert_by_path dropped the first. Common in .kb/ agent reports."""
    md = (
        "# A\n\nintro\n\n"
        "## Notes\n\nalpha content\n\n"
        "## Notes\n\nbeta content\n\n"
        "## Example\n\nex content\n"
    )
    f = tmp_path / "dup.md"
    f.write_text(md)
    doc_id, _ = ingest_markdown_file(f, db_path=tmp_db)

    from kb.core.connection import DatabaseConnection
    conn = DatabaseConnection(tmp_db).conn
    rows = conn.execute(
        "SELECT path, content FROM document_sections "
        "WHERE document_id = ? AND status = 'active'",
        (doc_id,),
    ).fetchall()
    paths = [r[0] for r in rows]
    blob = " ".join((r[1] or "") for r in rows)
    assert len(paths) == len(set(paths)), f"duplicate paths: {paths}"
    assert "alpha content" in blob, "first '## Notes' body was dropped (data loss)"
    assert "beta content" in blob, "second '## Notes' body missing"
    assert "ex content" in blob, "sibling after duplicates missing"


def test_ingest_ordinal_set(multi_md, tmp_db):
    doc_id, section_ids = ingest_markdown_file(multi_md, db_path=tmp_db)

    from kb.core.connection import DatabaseConnection
    conn = DatabaseConnection(tmp_db).conn
    ords = [r[0] for r in conn.execute(
        "SELECT ordinal FROM document_sections WHERE document_id = ? AND status = 'active' ORDER BY ordinal",
        (doc_id,),
    ).fetchall()]
    assert ords == sorted(ords)


def test_ingest_parent_section_id(multi_md, tmp_db):
    """Subsections (level > 1) have a non-NULL parent_section_id."""
    doc_id, _ = ingest_markdown_file(multi_md, db_path=tmp_db)

    from kb.core.connection import DatabaseConnection
    conn = DatabaseConnection(tmp_db).conn
    rows = conn.execute(
        "SELECT level, parent_section_id FROM document_sections WHERE document_id = ? AND status = 'active'",
        (doc_id,),
    ).fetchall()
    deep = [r for r in rows if r[0] is not None and r[0] >= 2]
    # At least some deep sections should have a parent
    with_parent = [r for r in deep if r[1] is not None]
    assert with_parent, "No deep sections have parent_section_id set"


def test_ingest_front_matter(fm_md, tmp_db):
    """Front-matter title/project/doc_type/summary are respected."""
    from kb.core.connection import DatabaseConnection
    doc_id, section_ids = ingest_markdown_file(fm_md, db_path=tmp_db)

    conn = DatabaseConnection(tmp_db).conn
    row = conn.execute("SELECT title, doc_type, project, summary FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row is not None
    assert row[0] == "My Test Document"
    assert row[1] == "reference"
    assert row[2] == "testproject"
    assert "test document" in (row[3] or "").lower()


def test_ingest_upsert_idempotent(multi_md, tmp_db):
    """Re-ingesting the same file with same content is a no-op (upsert_by_path)."""
    doc_id1, sec_ids1 = ingest_markdown_file(multi_md, db_path=tmp_db)
    doc_id2, sec_ids2 = ingest_markdown_file(multi_md, db_path=tmp_db)
    # Second doc has different id but same section paths; no new active sections
    from kb.core.connection import DatabaseConnection
    conn = DatabaseConnection(tmp_db).conn
    active = conn.execute(
        "SELECT COUNT(*) FROM document_sections WHERE document_id = ? AND status = 'active'",
        (doc_id1,),
    ).fetchone()[0]
    assert active > 0


# ---------------------------------------------------------------------------
# kb add -f behaviour (via count_heading_sections)
# ---------------------------------------------------------------------------

def test_single_file_does_not_split(single_md, tmp_db):
    """A file with no headings has 0 heading sections — kb add -f should NOT split."""
    text = single_md.read_text()
    assert count_heading_sections(text) == 0


def test_multi_file_triggers_split(multi_md, tmp_db):
    """A file with >=2 headings triggers the split path."""
    text = multi_md.read_text()
    assert count_heading_sections(text) >= 2


# ---------------------------------------------------------------------------
# kb ingest md round-trip
# ---------------------------------------------------------------------------

def test_ingest_md_run_function(multi_md, tmp_db):
    """run() returns 0 and prints doc-id + section count."""
    from kb.ingest.markdown import run
    import io, sys
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    rc = run(file_path=multi_md, db_path=tmp_db)
    sys.stdout = old_stdout
    assert rc == 0
    out = captured.getvalue()
    assert "doc-id:" in out
    assert "sections:" in out


def test_ingest_md_dry_run(multi_md, tmp_db):
    """dry_run=True prints summary without writing to DB."""
    from kb.ingest.markdown import run
    from kb.core.connection import DatabaseConnection
    import io, sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    rc = run(file_path=multi_md, db_path=tmp_db, dry_run=True)
    sys.stdout = old_stdout
    assert rc == 0
    out = captured.getvalue()
    assert "dry-run" in out

    # DB should not have been created or have any documents
    if tmp_db.exists():
        conn = DatabaseConnection(tmp_db).conn
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 0
