"""
PDF document ingest — fitz outline + docling front-end.

Uses PyMuPDF (fitz) for the section outline and docling for reading-order text
and table structure.  Both are OPTIONAL extras; import errors produce a clear
installation message pointing to ``pip install 'kb[pdf]'``.

Public API
----------
ingest_pdf_file(path, db_path, project, doc_type)
    -> (doc_id, [section_ids])

run(file_path, db_path, project, doc_type, title, summary, dry_run)
    -> int (exit code)
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import DEFAULT_DB_PATH
from kb.entities.documents import DocumentsRepository
from kb.entities.document_sections import DocumentSectionsRepository
from kb.core.connection import DatabaseConnection
from kb.core.schema import init_schema
from kb.ingest.markdown import (
    TOKEN_BUDGET,
    WINDOW_OVERLAP,
    _content_hash,
    _token_count,
    _window_split,
    _linearize_table,
    _slug,
    _build_parent_map,
)


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

_INSTALL_MSG = (
    "PDF ingest requires docling and pymupdf. "
    "Install them with: pip install 'kb[pdf]'\n"
    "(docling pulls CUDA torch; use a separate venv with system-site-packages "
    "for CUDA torch, or CPU-only: pip install docling pymupdf)"
)


def _require_pdf_deps():
    """Import fitz and docling; raise ImportError with install hint if missing."""
    try:
        import fitz  # noqa: F401 — PyMuPDF
    except ImportError as e:
        raise ImportError(_INSTALL_MSG) from e
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: F401
        from docling.datamodel.base_models import InputFormat  # noqa: F401
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_MSG) from e


# ---------------------------------------------------------------------------
# Scanned-PDF detection
# ---------------------------------------------------------------------------

_SCANNED_SAMPLE_PAGES = 5   # number of pages to sample for text
_SCANNED_MIN_CHARS = 20     # fewer chars across sample → treat as scanned


def _is_scanned(fitz_doc) -> bool:
    """Return True if the PDF has no embedded text layer (scanned document).

    Samples up to _SCANNED_SAMPLE_PAGES pages evenly across the document.
    If total extracted text across the sample is below _SCANNED_MIN_CHARS the
    document is considered scanned.
    """
    n = len(fitz_doc)
    if n == 0:
        return True
    step = max(1, n // _SCANNED_SAMPLE_PAGES)
    indices = list(range(0, n, step))[:_SCANNED_SAMPLE_PAGES]
    total_chars = sum(
        len(fitz_doc[i].get_text("text").strip()) for i in indices
    )
    return total_chars < _SCANNED_MIN_CHARS


# ---------------------------------------------------------------------------
# fitz outline → section tree
# ---------------------------------------------------------------------------

def _fitz_outline_to_tree(toc: list) -> list[dict[str, Any]]:
    """Convert fitz get_toc() output to a list of outline nodes.

    Each node: {level, heading, page (0-based)}.
    fitz returns [(level, title, page), ...] with pages 1-based.
    """
    return [
        {"level": lvl, "heading": title.strip(), "page": max(0, page - 1)}
        for lvl, title, page in toc
        if title and title.strip()
    ]


# ---------------------------------------------------------------------------
# Table HTML export and linearization
# ---------------------------------------------------------------------------

def _table_to_html(tbl, doc) -> str:
    """Export a docling TableItem to HTML string."""
    try:
        return tbl.export_to_html(doc)
    except Exception:
        # Fallback: export to markdown if HTML export unavailable
        try:
            return tbl.export_to_markdown(doc)
        except Exception:
            return ""


def _linearize_html_table(html: str) -> str:
    """Linearize an HTML table to row-per-line text for embedding.

    Each row becomes: ``cell1 | cell2 | cell3``
    Strips HTML tags; skips empty rows.
    """
    # Extract text from each <td>/<th> block; group by rows.
    row_pat = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pat = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    tag_pat = re.compile(r'<[^>]+>')

    rows: list[str] = []
    for row_m in row_pat.finditer(html):
        cells = [tag_pat.sub('', c.group(1)).strip()
                 for c in cell_pat.finditer(row_m.group(1))]
        cells = [c.replace('\n', ' ') for c in cells if c]
        if cells:
            rows.append(' | '.join(cells))
    if rows:
        return '\n'.join(rows)
    # Fallback: strip all tags and return plain text
    return tag_pat.sub(' ', html).strip()


# ---------------------------------------------------------------------------
# Build intermediate from docling DoclingDocument + fitz outline
# ---------------------------------------------------------------------------

def _build_intermediate_pdf(
    doc,          # docling DoclingDocument
    outline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map a docling DoclingDocument → common intermediate.

    Shape of each entry (same as markdown.py _build_intermediate output):
      level, heading, raw_body, kind, content, embed_text, table_repr,
      is_interior, ordinal, path (computed here via outline alignment)

    Strategy:
    - If the fitz outline is present (non-empty), use it as the authoritative
      section tree.  We assign each docling item to its enclosing outline section
      by page number.
    - If no outline, use docling's own heading items to build the tree.
    - Tables become kind='table' leaves with HTML content + linearized embed_text.
    - Figures become kind='figure' leaves with caption.
    - Prose sections: collected per heading, window-split if oversize.
    """
    from docling.datamodel.document import DoclingDocument  # type: ignore
    # Import item types conditionally (API varies by version)
    try:
        from docling.datamodel.document import (
            TableItem, TextItem, SectionHeaderItem, PictureItem,
        )
        _have_item_types = True
    except ImportError:
        _have_item_types = False

    intermediate: list[dict[str, Any]] = []
    ordinal = 0

    # -----------------------------------------------------------------------
    # Build the heading tree from outline or docling headers
    # -----------------------------------------------------------------------
    if outline:
        # Use fitz outline.  Each section spans from its page to the next
        # outline entry's page (exclusive).  Assign docling body items by page.
        sections: list[dict[str, Any]] = []
        for i, node in enumerate(outline):
            next_page = outline[i + 1]["page"] if i + 1 < len(outline) else None
            sections.append({
                "level": node["level"],
                "heading": node["heading"],
                "page_start": node["page"],
                "page_end": next_page,
                "prose_parts": [],
                "tables": [],
                "figures": [],
            })
    else:
        # Fall back to docling's section headers
        sections = []
        if _have_item_types:
            for item, _ in doc.iterate_items():
                if isinstance(item, SectionHeaderItem):
                    sections.append({
                        "level": getattr(item, "level", 1),
                        "heading": item.text or "",
                        "page_start": _item_page(item),
                        "page_end": None,
                        "prose_parts": [],
                        "tables": [],
                        "figures": [],
                    })
        # Set page_end from next section's page_start
        for i in range(len(sections) - 1):
            sections[i]["page_end"] = sections[i + 1]["page_start"]

    # Add a catch-all section for content before the first heading
    preamble: dict[str, Any] = {
        "level": 0,
        "heading": "",
        "page_start": 0,
        "page_end": sections[0]["page_start"] if sections else None,
        "prose_parts": [],
        "tables": [],
        "figures": [],
    }

    def _assign_section(page: int | None) -> dict[str, Any]:
        if page is None:
            return preamble
        # Find the deepest section that starts on or before this page
        best = preamble
        for sec in sections:
            if sec["page_start"] <= page:
                if sec["page_end"] is None or page < sec["page_end"]:
                    best = sec
        return best

    # -----------------------------------------------------------------------
    # Walk docling items and assign to sections
    # -----------------------------------------------------------------------
    # Collect all tables first so we can reference them
    tables_by_ref: dict[str, Any] = {}
    if hasattr(doc, 'tables'):
        for tbl in doc.tables:
            ref = id(tbl)
            tables_by_ref[ref] = tbl

    if _have_item_types:
        for item, _ in doc.iterate_items():
            page = _item_page(item)
            sec = _assign_section(page)

            if isinstance(item, SectionHeaderItem):
                # Already used for structure; skip body assignment
                continue
            elif isinstance(item, TableItem):
                sec["tables"].append(item)
            elif isinstance(item, PictureItem):
                caption = ""
                if hasattr(item, 'captions') and item.captions:
                    caption = " ".join(
                        c.text for c in item.captions if hasattr(c, 'text') and c.text
                    )
                sec["figures"].append(caption or "[figure]")
            elif isinstance(item, TextItem):
                text = (item.text or "").strip()
                if text:
                    sec["prose_parts"].append(text)
    else:
        # Fallback: no item-type discrimination; walk export_to_dict
        try:
            body = doc.export_to_markdown()
            preamble["prose_parts"].append(body)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Build intermediate entries from sections
    # -----------------------------------------------------------------------
    all_sections = [preamble] + sections

    # Level counters for path computation (mirrors markdown.py approach)
    level_counters = [0] * 7
    section_paths: list[str] = []

    for sec in all_sections:
        lvl = sec["level"]
        if lvl == 0:
            section_paths.append("0")
        else:
            level_counters[lvl] += 1
            for sub in range(lvl + 1, 7):
                level_counters[sub] = 0
            section_paths.append(
                ".".join(str(level_counters[l]) for l in range(1, lvl + 1))
            )

    for sec_idx, sec in enumerate(all_sections):
        base_path = section_paths[sec_idx]
        table_counter = 0
        fig_counter = 0

        # Prose
        prose_text = "\n\n".join(sec["prose_parts"]).strip()
        if prose_text:
            tcount = _token_count(prose_text)
            if tcount > TOKEN_BUDGET:
                parts = _window_split(prose_text, TOKEN_BUDGET, WINDOW_OVERLAP)
                n_parts = len(parts)
                for pi, part_text in enumerate(parts):
                    path = f"{base_path}.p{pi + 1}" if n_parts > 1 else base_path
                    intermediate.append({
                        "level": sec["level"],
                        "heading": sec["heading"],
                        "kind": "prose",
                        "content": part_text,
                        "embed_text": part_text,
                        "table_repr": None,
                        "is_interior": False,
                        "ordinal": ordinal,
                        "path": path,
                    })
                    ordinal += 1
            else:
                intermediate.append({
                    "level": sec["level"],
                    "heading": sec["heading"],
                    "kind": "prose",
                    "content": prose_text,
                    "embed_text": prose_text,
                    "table_repr": None,
                    "is_interior": False,
                    "ordinal": ordinal,
                    "path": base_path,
                })
                ordinal += 1
        elif not sec["tables"] and not sec["figures"]:
            # Empty section (heading only) — still emit so heading node exists
            intermediate.append({
                "level": sec["level"],
                "heading": sec["heading"],
                "kind": "prose",
                "content": sec["heading"],
                "embed_text": sec["heading"],
                "table_repr": None,
                "is_interior": True,
                "ordinal": ordinal,
                "path": base_path,
            })
            ordinal += 1

        # Tables
        for tbl in sec["tables"]:
            table_counter += 1
            html = _table_to_html(tbl, doc)
            embed_text = _linearize_html_table(html) if html else ""
            path = f"{base_path}.t{table_counter}"
            intermediate.append({
                "level": sec["level"],
                "heading": sec["heading"],
                "kind": "table",
                "content": html,
                "embed_text": embed_text,
                "table_repr": html,
                "is_interior": False,
                "ordinal": ordinal,
                "path": path,
            })
            ordinal += 1

        # Figures
        for caption in sec["figures"]:
            fig_counter += 1
            path = f"{base_path}.f{fig_counter}"
            embed_text = f"Figure: {caption}"
            intermediate.append({
                "level": sec["level"],
                "heading": sec["heading"],
                "kind": "figure",
                "content": caption,
                "embed_text": embed_text,
                "table_repr": None,
                "is_interior": False,
                "ordinal": ordinal,
                "path": path,
            })
            ordinal += 1

    return intermediate, all_sections, section_paths


def _item_page(item) -> int | None:
    """Extract 0-based page number from a docling item, or None."""
    try:
        prov = item.prov
        if prov:
            return prov[0].page_no - 1  # docling is 1-based
    except (AttributeError, IndexError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Persistence helper (mirrors markdown.py logic, shared path)
# ---------------------------------------------------------------------------

def _persist_intermediate(
    intermediate: list[dict[str, Any]],
    all_sections: list[dict[str, Any]],
    section_paths: list[str],
    doc_id: str,
    secs_repo: DocumentSectionsRepository,
) -> list[str]:
    """Persist intermediate entries to document_sections.

    Returns list of section_ids in ordinal order (leaf sections).
    """
    # Build a parent map using level hierarchy
    raw_for_parent = [{"level": s["level"], "heading": s["heading"]} for s in all_sections]
    parent_map = _build_parent_map(raw_for_parent)

    # Insert one heading node per real section (level >= 1)
    raw_section_db_ids: dict[int, str] = {}
    for si, sec in enumerate(all_sections):
        if sec["level"] == 0:
            continue
        parent_si = parent_map.get(si)
        parent_db_id = raw_section_db_ids.get(parent_si) if parent_si is not None else None
        heading_path = section_paths[si]

        section_id, _ = secs_repo.upsert_by_path(
            document_id=doc_id,
            path=heading_path,
            content_hash=_content_hash(sec["heading"]),
            level=sec["level"],
            ordinal=si,
            kind="prose",
            heading=sec["heading"],
            content=sec["heading"],
            embed_text=sec["heading"],
            parent_section_id=parent_db_id,
        )
        raw_section_db_ids[si] = section_id

    # Insert leaf entries
    section_ids: list[str] = []
    for entry in intermediate:
        # Find which all_sections entry this belongs to by (level, heading, path-prefix)
        entry_base = re.sub(r'\.(t\d+|p\d+|f\d+)$', '', entry["path"])
        parent_db_id = None
        for si, sp in enumerate(section_paths):
            if sp == entry_base and all_sections[si]["level"] >= 1:
                parent_db_id = raw_section_db_ids.get(si)
                break

        section_id, _ = secs_repo.upsert_by_path(
            document_id=doc_id,
            path=entry["path"],
            content_hash=_content_hash(entry["content"] or ""),
            level=entry["level"],
            ordinal=entry["ordinal"],
            kind=entry["kind"],
            heading=entry["heading"] or None,
            content=entry["content"],
            embed_text=entry.get("embed_text") or entry["content"],
            table_repr=entry.get("table_repr"),
            parent_section_id=parent_db_id,
        )
        section_ids.append(section_id)

    return section_ids


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def ingest_pdf_file(
    path: Path,
    db_path: Path | None = None,
    project: str | None = None,
    doc_type: str | None = None,
    title: str | None = None,
    summary: str | None = None,
) -> tuple[str, list[str]]:
    """Ingest a PDF into document + sections.

    Steps:
    1. PyMuPDF: detect scanned PDF; extract outline (TOC).
    2. If scanned (no text layer): raise NotImplementedError → kb-0f67bf fallback.
    3. docling (do_ocr=False, FAST): reading order, tables as HTML, figures.
    4. Align docling output to fitz outline (or use docling headers if no outline).
    5. Persist via DocumentsRepository + DocumentSectionsRepository.

    Returns (doc_id, [section_ids]).
    """
    _require_pdf_deps()

    import fitz  # PyMuPDF
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    path = Path(path).expanduser().resolve()

    # --- Step 1: fitz ---
    fitz_doc = fitz.open(str(path))

    # --- Step 2: Scanned detection ---
    if _is_scanned(fitz_doc):
        fitz_doc.close()
        raise NotImplementedError(
            f"PDF appears to be a scanned document (no embedded text layer): {path}\n"
            "PaddleOCR-VL scanned-PDF fallback is tracked in kb-0f67bf and not yet implemented.\n"
            "To ingest scanned PDFs, use a version of kb that includes the OCR backend."
        )

    toc = fitz_doc.get_toc()
    outline = _fitz_outline_to_tree(toc)
    fitz_doc.close()

    # --- Step 3: docling ---
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(path))
    docling_doc = result.document

    # --- Step 4: Build intermediate ---
    intermediate, all_sections, section_paths = _build_intermediate_pdf(
        docling_doc, outline
    )

    # --- Step 5: Persist ---
    raw_bytes = path.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()

    if doc_type is None:
        doc_type = "reference"
    if title is None:
        title = path.stem.replace("-", " ").replace("_", " ")

    from kb.config import load_config
    cfg = load_config()
    resolved_db = Path(db_path) if db_path else cfg.db_path
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    db_conn = DatabaseConnection(resolved_db, cfg.embedding_dim)
    conn = db_conn.conn
    init_schema(conn, cfg.embedding_dim)

    docs_repo = DocumentsRepository(conn)
    secs_repo = DocumentSectionsRepository(conn)

    doc_id = docs_repo.add(
        title=title,
        doc_type=doc_type,
        source_path=str(path),
        source_hash=source_hash,
        project=project,
        summary=summary,
    )

    section_ids = _persist_intermediate(
        intermediate, all_sections, section_paths, doc_id, secs_repo
    )

    return doc_id, section_ids


# ---------------------------------------------------------------------------
# CLI entry point (kb ingest pdf <file>)
# ---------------------------------------------------------------------------

def run(
    file_path: Path,
    db_path: Path | None = None,
    project: str | None = None,
    doc_type: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    dry_run: bool = False,
) -> int:
    """Ingest a PDF file. Returns 0 on success."""
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    if dry_run:
        try:
            _require_pdf_deps()
            import fitz
            fitz_doc = fitz.open(str(file_path))
            toc = fitz_doc.get_toc()
            is_scanned = _is_scanned(fitz_doc)
            n_pages = len(fitz_doc)
            fitz_doc.close()
            print(f"[dry-run] {file_path.name}: {n_pages} pages, "
                  f"{len(toc)} outline entries, scanned={is_scanned}")
            for lvl, ttl, pg in toc[:20]:
                print(f"  {'  ' * (lvl - 1)}[{lvl}] {ttl} (p{pg})")
            if len(toc) > 20:
                print(f"  ... ({len(toc) - 20} more entries)")
        except ImportError as e:
            print(f"[dry-run] {file_path.name}: (pdf deps unavailable: {e})")
        return 0

    try:
        doc_id, section_ids = ingest_pdf_file(
            file_path,
            db_path=db_path,
            project=project,
            doc_type=doc_type,
            title=title,
            summary=summary,
        )
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}", file=sys.stderr)
        return 1

    print(f"doc-id: {doc_id}")
    print(f"sections: {len(section_ids)}")
    for sid in section_ids:
        print(f"  {sid}")
    return 0
