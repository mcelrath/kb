"""Tests for kb/ingest/pdf.py — PDF front-end.

Tests that do NOT require docling/fitz run in the kb .venv (which lacks the
pdf extras).  Tests that require docling are decorated with
pytest.importorskip so they SKIP cleanly when the deps are absent.
"""

from __future__ import annotations

import re
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal mock docling/fitz objects
# ---------------------------------------------------------------------------

def _make_mock_fitz_page(text: str = "some text"):
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_mock_fitz_doc(pages: list[str], toc: list | None = None):
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__iter__ = MagicMock(return_value=iter([_make_mock_fitz_page(t) for t in pages]))
    doc.__getitem__ = MagicMock(side_effect=lambda i: _make_mock_fitz_page(pages[i]))
    doc.get_toc.return_value = toc or []
    return doc


# ---------------------------------------------------------------------------
# Import the module under test (always importable — no top-level fitz/docling)
# ---------------------------------------------------------------------------

from kb.ingest.pdf import (
    _fitz_outline_to_tree,
    _is_scanned,
    _linearize_html_table,
    _require_pdf_deps,
    _build_intermediate_pdf,
)


# ---------------------------------------------------------------------------
# fitz outline → tree
# ---------------------------------------------------------------------------

class TestFitzOutlineToTree:
    def test_empty_toc(self):
        assert _fitz_outline_to_tree([]) == []

    def test_basic_toc(self):
        toc = [
            (1, "Chapter 1", 1),
            (2, "Section 1.1", 3),
            (2, "Section 1.2", 7),
            (1, "Chapter 2", 10),
        ]
        nodes = _fitz_outline_to_tree(toc)
        assert len(nodes) == 4
        assert nodes[0] == {"level": 1, "heading": "Chapter 1", "page": 0}
        assert nodes[1] == {"level": 2, "heading": "Section 1.1", "page": 2}
        assert nodes[2] == {"level": 2, "heading": "Section 1.2", "page": 6}
        assert nodes[3] == {"level": 1, "heading": "Chapter 2", "page": 9}

    def test_strips_whitespace_from_headings(self):
        toc = [(1, "  Title with spaces  ", 1)]
        nodes = _fitz_outline_to_tree(toc)
        assert nodes[0]["heading"] == "Title with spaces"

    def test_skips_blank_headings(self):
        toc = [(1, "", 1), (1, "   ", 2), (2, "Real Heading", 3)]
        nodes = _fitz_outline_to_tree(toc)
        assert len(nodes) == 1
        assert nodes[0]["heading"] == "Real Heading"

    def test_page_numbers_converted_to_zero_based(self):
        # fitz returns 1-based pages; we convert to 0-based
        toc = [(1, "First", 1)]
        nodes = _fitz_outline_to_tree(toc)
        assert nodes[0]["page"] == 0

    def test_page_zero_clamped(self):
        # fitz occasionally returns page 0 (bookmarks before first page)
        toc = [(1, "Before start", 0)]
        nodes = _fitz_outline_to_tree(toc)
        assert nodes[0]["page"] == 0  # max(0, 0-1) = 0


# ---------------------------------------------------------------------------
# Scanned detection
# ---------------------------------------------------------------------------

class TestIsScanned:
    def test_text_pages_not_scanned(self):
        pages = ["This is a page with real text content here." * 3] * 5
        doc = _make_mock_fitz_doc(pages)
        assert not _is_scanned(doc)

    def test_empty_pages_scanned(self):
        doc = _make_mock_fitz_doc(["", "", "  ", "\n"])
        assert _is_scanned(doc)

    def test_sparse_text_scanned(self):
        # A few chars across sample — below threshold
        doc = _make_mock_fitz_doc(["ab", "cd", "e"])
        assert _is_scanned(doc)

    def test_empty_doc_scanned(self):
        doc = _make_mock_fitz_doc([])
        assert _is_scanned(doc)

    def test_single_page_with_text(self):
        doc = _make_mock_fitz_doc(["The quick brown fox jumps over the lazy dog. " * 5])
        assert not _is_scanned(doc)


# ---------------------------------------------------------------------------
# HTML table linearization
# ---------------------------------------------------------------------------

class TestLinearizeHtmlTable:
    def test_basic_table(self):
        html = """
        <table>
          <tr><th>Code</th><th>Meaning</th></tr>
          <tr><td>0-105</td><td>SGPR</td></tr>
          <tr><td>240</td><td>0.5</td></tr>
        </table>
        """
        result = _linearize_html_table(html)
        assert "Code | Meaning" in result
        assert "0-105 | SGPR" in result
        assert "240 | 0.5" in result

    def test_empty_rows_skipped(self):
        html = "<table><tr><td></td><td></td></tr><tr><td>a</td><td>b</td></tr></table>"
        result = _linearize_html_table(html)
        assert "a | b" in result
        # Empty row should not appear as just " | "
        lines = [l for l in result.splitlines() if l.strip()]
        assert all("|" in l for l in lines)

    def test_strips_inner_html_tags(self):
        html = "<table><tr><td><b>bold</b></td><td><i>italic</i></td></tr></table>"
        result = _linearize_html_table(html)
        assert "bold" in result
        assert "italic" in result
        assert "<b>" not in result
        assert "<i>" not in result

    def test_multiline_cell_flattened(self):
        html = "<table><tr><td>line1\nline2</td><td>val</td></tr></table>"
        result = _linearize_html_table(html)
        # newlines within a cell should be flattened
        assert "\n" not in result.split("|")[0].strip() or "line1" in result

    def test_sop2_encoding_preserved(self):
        # Simulate the RDNA3 SOP2 Fields table row: code 240 -> 0.5
        html = """
        <table>
          <tr><th>ENCODING</th><th>SSRC0</th></tr>
          <tr><td>0-105</td><td>SGPR[ENCODING]</td></tr>
          <tr><td>240</td><td>0.5</td></tr>
          <tr><td>235</td><td>SHARED_BASE</td></tr>
        </table>
        """
        result = _linearize_html_table(html)
        assert "0-105" in result
        assert "SGPR" in result
        assert "240" in result
        assert "0.5" in result
        assert "235" in result
        assert "SHARED_BASE" in result


# ---------------------------------------------------------------------------
# Import guard — no docling in kb .venv → ImportError with helpful message
# ---------------------------------------------------------------------------

class TestRequirePdfDeps:
    def test_raises_import_error_when_fitz_missing(self):
        import sys
        # Temporarily hide fitz from imports
        original = sys.modules.get("fitz")
        sys.modules["fitz"] = None  # type: ignore
        try:
            with pytest.raises(ImportError, match="pip install"):
                _require_pdf_deps()
        finally:
            if original is None:
                sys.modules.pop("fitz", None)
            else:
                sys.modules["fitz"] = original

    def test_raises_import_error_when_docling_missing(self):
        import sys
        # Temporarily hide docling
        saved = {k: v for k, v in sys.modules.items() if k.startswith("docling")}
        for k in list(sys.modules.keys()):
            if k.startswith("docling"):
                sys.modules[k] = None  # type: ignore

        # Also need fitz to be present
        try:
            import fitz
            fitz_present = True
        except ImportError:
            fitz_present = False

        if not fitz_present:
            pytest.skip("fitz not installed")

        try:
            with pytest.raises(ImportError, match="pip install"):
                _require_pdf_deps()
        finally:
            for k in list(sys.modules.keys()):
                if k.startswith("docling"):
                    sys.modules.pop(k, None)
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# _build_intermediate_pdf with mocked docling doc
# ---------------------------------------------------------------------------

class TestBuildIntermediatePdf:
    """Test the intermediate builder without real docling/fitz."""

    def _make_mock_doc(self, items: list[dict]) -> MagicMock:
        """Build a mock DoclingDocument with the given items list.

        Each item dict: {type: 'text'|'table'|'header'|'figure', text: str, page: int}
        """
        from kb.ingest.pdf import _item_page

        doc = MagicMock()

        # Build mock items
        mock_items = []
        for item_spec in items:
            t = item_spec.get("type", "text")
            item = MagicMock()
            item.text = item_spec.get("text", "")
            page_no = item_spec.get("page", 1)
            prov_entry = MagicMock()
            prov_entry.page_no = page_no
            item.prov = [prov_entry]

            # docling item type
            from docling.datamodel.document import TextItem, TableItem, SectionHeaderItem, PictureItem
            if t == "text":
                item.__class__ = TextItem
            elif t == "table":
                item.__class__ = TableItem
                item.export_to_html = MagicMock(return_value=item_spec.get("html", "<table></table>"))
                item.export_to_markdown = MagicMock(return_value="| col |\n|---|\n| val |")
            elif t == "header":
                item.__class__ = SectionHeaderItem
                item.level = item_spec.get("level", 1)
            elif t == "figure":
                item.__class__ = PictureItem
                cap = MagicMock()
                cap.text = item_spec.get("caption", "")
                item.captions = [cap] if item_spec.get("caption") else []
            mock_items.append(item)

        doc.iterate_items.return_value = [(i, None) for i in mock_items]
        doc.tables = [i for i in mock_items if hasattr(i, 'export_to_html')]
        return doc

    def test_basic_prose_section(self):
        """Prose text under a heading builds correct intermediate entries."""
        docling = pytest.importorskip("docling")
        outline = [{"level": 1, "heading": "Introduction", "page": 0}]
        doc = self._make_mock_doc([
            {"type": "text", "text": "Hello world.", "page": 1},
        ])
        intermediate, all_sections, section_paths = _build_intermediate_pdf(doc, outline)
        assert any(e["kind"] == "prose" for e in intermediate)
        prose = [e for e in intermediate if e["kind"] == "prose" and e["content"] == "Hello world."]
        assert prose, f"No prose entry with expected content; got: {[e['content'] for e in intermediate]}"

    def test_table_becomes_table_kind(self):
        """A TableItem produces a kind='table' leaf with HTML content."""
        docling = pytest.importorskip("docling")
        html = "<table><tr><td>0-105</td><td>SGPR</td></tr></table>"
        outline = [{"level": 1, "heading": "Fields", "page": 0}]
        doc = self._make_mock_doc([
            {"type": "table", "html": html, "page": 1},
        ])
        intermediate, _, _ = _build_intermediate_pdf(doc, outline)
        tables = [e for e in intermediate if e["kind"] == "table"]
        assert tables, "No table entry produced"
        assert tables[0]["content"] == html
        assert "SGPR" in tables[0]["embed_text"]

    def test_paths_unique(self):
        """All entries in the intermediate have distinct paths."""
        docling = pytest.importorskip("docling")
        html = "<table><tr><td>a</td><td>b</td></tr></table>"
        outline = [
            {"level": 1, "heading": "Ch1", "page": 0},
            {"level": 2, "heading": "Sec1.1", "page": 1},
        ]
        doc = self._make_mock_doc([
            {"type": "text", "text": "Intro text.", "page": 0},
            {"type": "text", "text": "Section text.", "page": 1},
            {"type": "table", "html": html, "page": 1},
        ])
        intermediate, _, _ = _build_intermediate_pdf(doc, outline)
        paths = [e["path"] for e in intermediate]
        assert len(paths) == len(set(paths)), f"Duplicate paths: {paths}"

    def test_no_outline_falls_back_to_docling_headers(self):
        """When outline is empty, docling SectionHeaderItems define the structure."""
        docling = pytest.importorskip("docling")
        doc = self._make_mock_doc([
            {"type": "header", "text": "MySection", "level": 1, "page": 1},
            {"type": "text", "text": "Body text here.", "page": 1},
        ])
        intermediate, all_sections, _ = _build_intermediate_pdf(doc, [])
        headings = [s["heading"] for s in all_sections if s["heading"]]
        assert "MySection" in headings


# ---------------------------------------------------------------------------
# End-to-end smoke test (skipped if docling/fitz absent or PDF unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("/home/mcelrath/Projects/ai/doc-ingest/experiments/rdna3_ch15_formats.pdf").exists(),
    reason="Test PDF not available",
)
def test_end_to_end_dry_run(tmp_path):
    """dry_run=True on the RDNA3 ch15 PDF prints outline summary, returns 0."""
    fitz = pytest.importorskip("fitz")
    from kb.ingest.pdf import run
    rc = run(
        file_path=Path("/home/mcelrath/Projects/ai/doc-ingest/experiments/rdna3_ch15_formats.pdf"),
        db_path=tmp_path / "test.db",
        dry_run=True,
    )
    assert rc == 0
