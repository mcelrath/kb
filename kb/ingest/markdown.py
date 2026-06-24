"""
Markdown document ingest — heading-tree chunker.

Parses ATX (# ... ######) and setext (underline) headings into a section tree,
persists one DocumentsRepository document + N DocumentSectionsRepository sections.

Public API
----------
ingest_markdown_file(file_path, db_path, doc_type, project, title, summary)
    -> (doc_id, [section_ids])

count_heading_sections(text)
    -> int   (number of heading-bounded sections; 0 if no headings)
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

# Package root on sys.path so this module can be used standalone.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import DEFAULT_DB_PATH
from kb.entities.documents import DocumentsRepository
from kb.entities.document_sections import DocumentSectionsRepository
from kb.core.connection import DatabaseConnection
from kb.core.schema import init_schema


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_BUDGET = 1200   # rough word-count proxy; 1 token ≈ 1 word for these purposes
WINDOW_OVERLAP = 100  # overlap words for windowed split

_ATX_RE = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$')
_SETEXT_H1 = re.compile(r'^={3,}\s*$')
_SETEXT_H2 = re.compile(r'^-{3,}\s*$')
_TABLE_ROW = re.compile(r'^\s*\|')
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]*)\)')
_FRONT_MATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


# ---------------------------------------------------------------------------
# Front-matter parser
# ---------------------------------------------------------------------------

def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML front-matter (if present) and return (meta, body)."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip()
    return meta, body


# ---------------------------------------------------------------------------
# Heading tree parser
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Rough token count (word split — good enough for budget gating)."""
    return len(text.split())


def _slug(heading: str) -> str:
    """Slugify a heading for path segments."""
    return re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _parse_sections(text: str) -> list[dict[str, Any]]:
    """Parse markdown text into a flat list of heading-bounded sections.

    Each entry:
      level     : int (1-6)
      heading   : str
      raw_body  : str (text between this heading and the next)

    A synthetic level-0 entry (heading='') is prepended for any text that
    precedes the first real heading (intro / preamble).
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, str, int]] = []  # (level, heading, line_index)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        # ATX heading
        m = _ATX_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            sections.append((level, heading, i))
            i += 1
            continue
        # Setext heading: current line is heading, next is underline
        if i + 1 < len(lines):
            next_line = lines[i + 1].rstrip('\n')
            if _SETEXT_H1.match(next_line):
                sections.append((1, line.strip(), i))
                i += 2
                continue
            if _SETEXT_H2.match(next_line):
                sections.append((2, line.strip(), i))
                i += 2
                continue
        i += 1

    # Build result list with body text
    result: list[dict[str, Any]] = []
    n = len(sections)

    # Text before the first heading (level 0, empty heading)
    if sections:
        preamble_lines = lines[: sections[0][2]]
    else:
        preamble_lines = lines

    preamble = "".join(preamble_lines).strip()
    if preamble:
        result.append({"level": 0, "heading": "", "raw_body": preamble})

    for idx, (level, heading, line_idx) in enumerate(sections):
        body_start = line_idx + 1
        # For setext, body starts 2 lines after (heading + underline already consumed)
        body_end = sections[idx + 1][2] if idx + 1 < n else len(lines)
        body = "".join(lines[body_start:body_end]).strip()
        result.append({"level": level, "heading": heading, "raw_body": body})

    return result


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------

def _classify_body(body: str) -> str:
    """Classify a section body as 'prose', 'table', or 'figure'."""
    stripped = body.strip()
    lines = stripped.splitlines()
    # Figure: section is just an image
    img_match = _IMAGE_RE.search(stripped)
    if img_match and len(lines) <= 3:
        return "figure"
    # Table: majority of non-empty lines look like table rows
    non_empty = [l for l in lines if l.strip()]
    if non_empty:
        table_lines = sum(1 for l in non_empty if _TABLE_ROW.match(l))
        if table_lines / len(non_empty) >= 0.6:
            return "table"
    return "prose"


def _extract_tables(body: str) -> list[tuple[str, int, int]]:
    """Extract table blocks from a body; return list of (table_md, start_line, end_line).

    start_line/end_line index into body.splitlines(keepends=True), so callers strip
    a table by its LINE SPAN rather than by string match — a string match would delete
    every identical-looking block, including legitimately repeated table text.
    """
    result: list[tuple[str, int, int]] = []
    lines = body.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if _TABLE_ROW.match(lines[i]):
            start = i
            while i < len(lines) and (_TABLE_ROW.match(lines[i]) or lines[i].strip() == ''):
                if not _TABLE_ROW.match(lines[i]) and lines[i].strip() == '':
                    # Allow one blank line inside a table block
                    if i + 1 < len(lines) and _TABLE_ROW.match(lines[i + 1]):
                        i += 1
                        continue
                    break
                i += 1
            table_block = "".join(lines[start:i]).strip()
            if table_block:
                result.append((table_block, start, i))
        else:
            i += 1
    return result


def _linearize_table(table_md: str) -> str:
    """Convert a markdown table to a linearized row-per-line text shadow for embedding."""
    rows: list[str] = []
    for line in table_md.splitlines():
        line = line.strip()
        if not line or re.match(r'^\|[-| :]+\|$', line):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(' | '.join(cells))
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Window split for oversize leaves
# ---------------------------------------------------------------------------

def _window_split(text: str, budget: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of at most `budget` words."""
    words = text.split()
    if len(words) <= budget:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + budget, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Build intermediate section list
# ---------------------------------------------------------------------------

def _build_intermediate(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the flat intermediate list consumed by the persister.

    Each entry:
      level, heading, raw_body,
      kind, content, embed_text, table_repr,
      is_interior (True = heading node, not a leaf),
      ordinal (document-order index)
    """
    intermediate: list[dict[str, Any]] = []
    ordinal = 0

    for raw_idx, sec in enumerate(sections):
        level = sec["level"]
        heading = sec["heading"]
        body = sec["raw_body"]

        # Extract inline tables as separate table leaves
        table_spans = _extract_tables(body)
        table_blocks = [blk for blk, _s, _e in table_spans]
        # Strip tables from the prose body by LINE SPAN — string-replace would delete
        # every identical-looking block, including legitimately repeated table text.
        body_lines = body.splitlines(keepends=True)
        covered: set[int] = set()
        for _blk, s, e in table_spans:
            covered.update(range(s, e))
        prose_body = "".join(l for idx, l in enumerate(body_lines) if idx not in covered).strip()

        # Prose / figure classification on the de-tabled body
        if prose_body:
            kind = _classify_body(prose_body)
            if kind == "figure":
                img_m = _IMAGE_RE.search(prose_body)
                caption = img_m.group(1) if img_m else ""
                alt_url = img_m.group(2) if img_m else ""
                content = prose_body
                embed_text = f"Figure: {caption or alt_url}"
                intermediate.append({
                    "level": level,
                    "heading": heading,
                    "raw_body": body,
                    "kind": "figure",
                    "content": content,
                    "embed_text": embed_text,
                    "table_repr": None,
                    "is_interior": False,
                    "ordinal": ordinal,
                    "raw_idx": raw_idx,
                })
                ordinal += 1
            else:
                # Prose — may need window-split
                tcount = _token_count(prose_body)
                if tcount > TOKEN_BUDGET:
                    parts = _window_split(prose_body, TOKEN_BUDGET, WINDOW_OVERLAP)
                    n_parts = len(parts)
                    for part_idx, part_text in enumerate(parts):
                        intermediate.append({
                            "level": level,
                            "heading": heading,
                            "raw_body": part_text,
                            "kind": "prose",
                            "content": part_text,
                            "embed_text": part_text,
                            "table_repr": None,
                            "is_interior": False,
                            "ordinal": ordinal,
                            "part": (part_idx + 1, n_parts),
                            "raw_idx": raw_idx,
                        })
                        ordinal += 1
                else:
                    intermediate.append({
                        "level": level,
                        "heading": heading,
                        "raw_body": prose_body,
                        "kind": "prose",
                        "content": prose_body,
                        "embed_text": prose_body,
                        "table_repr": None,
                        "is_interior": False,
                        "ordinal": ordinal,
                        "raw_idx": raw_idx,
                    })
                    ordinal += 1

        # Table leaves
        for tb in table_blocks:
            linearized = _linearize_table(tb)
            intermediate.append({
                "level": level,
                "heading": heading,
                "raw_body": tb,
                "kind": "table",
                "content": tb,
                "embed_text": linearized,
                "table_repr": tb,
                "is_interior": False,
                "ordinal": ordinal,
                "raw_idx": raw_idx,
            })
            ordinal += 1

    return intermediate


# ---------------------------------------------------------------------------
# Path computation
# ---------------------------------------------------------------------------

def _raw_section_paths(raw_sections: list[dict[str, Any]]) -> list[str]:
    """Dotted-ordinal path per raw section (level-0 preamble -> '0').

    Heading counters reset at deeper levels, so two same-level siblings get
    distinct paths (1.1, 1.2) regardless of identical heading text.
    """
    level_counters: list[int] = [0] * 7  # index 0..6
    paths: list[str] = []
    for sec in raw_sections:
        lvl = sec["level"]
        if lvl == 0:
            paths.append("0")
            continue
        level_counters[lvl] += 1
        for sub in range(lvl + 1, 7):
            level_counters[sub] = 0
        paths.append(".".join(str(level_counters[l]) for l in range(1, lvl + 1)))
    return paths


def _doc_order_key(path: str) -> tuple:
    """Sortable document-order key for a section path.

    Paths like '1', '1.1', '1.1.t1', '1.5.3.f2' sort in true reading order:
    numeric parts compare as ints; table/part/figure suffixes (t/p/f) sort
    after the prose of their section. Used to assign a single monotonic `ordinal`
    across heading-nodes AND leaves so list/toc order is correct (the old code
    gave heading-nodes the section index and leaves a separate counter, which
    collided — 85 dup ordinals on RDNA3).
    """
    key: list[tuple] = []
    for part in path.split("."):
        if part.isdigit():
            key.append((0, int(part), ""))
        else:
            key.append((1, 0, part))  # t1/p1/f1 etc. — after numeric siblings
    return tuple(key)


def _ordinal_map(paths) -> dict[str, int]:
    """Map each (unique) section path to a 0-based document-order ordinal."""
    return {p: i for i, p in enumerate(sorted(set(paths), key=_doc_order_key))}


def _compute_paths(
    raw_sections: list[dict[str, Any]],
    intermediate: list[dict[str, Any]],
) -> None:
    """Annotate each intermediate entry with a stable path string.

    Paths are assigned via each entry's originating raw-section index
    (`raw_idx`, set in _build_intermediate) — NOT by re-matching (level,
    heading) text. The old text-match collided duplicate sibling headings
    (two '## Notes') onto one path and dropped content via upsert_by_path
    (kb-86b074). Tables get '<path>.t<N>', windowed prose parts '<path>.p<n>'.
    """
    raw_paths = _raw_section_paths(raw_sections)
    table_counter: dict[int, int] = {}  # raw_idx -> count

    for entry in intermediate:
        ri = entry["raw_idx"]
        base_path = raw_paths[ri] if 0 <= ri < len(raw_paths) else "0"
        kind = entry["kind"]
        part = entry.get("part")

        if kind == "table":
            cnt = table_counter.get(ri, 0) + 1
            table_counter[ri] = cnt
            entry["path"] = f"{base_path}.t{cnt}"
        elif part is not None:
            entry["path"] = f"{base_path}.p{part[0]}"
        else:
            entry["path"] = base_path


# ---------------------------------------------------------------------------
# Parent section ID mapping
# ---------------------------------------------------------------------------

def _build_parent_map(
    raw_sections: list[dict[str, Any]],
) -> dict[int, int | None]:
    """Return {raw_section_idx: parent_raw_section_idx | None}."""
    parent_map: dict[int, int | None] = {}
    stack: list[int] = []  # stack of raw_section indices

    for i, sec in enumerate(raw_sections):
        lvl = sec["level"]
        if lvl == 0:
            parent_map[i] = None
            continue
        # Pop stack until we find a section with smaller level
        while stack and raw_sections[stack[-1]]["level"] >= lvl:
            stack.pop()
        parent_map[i] = stack[-1] if stack else None
        stack.append(i)

    return parent_map


# ---------------------------------------------------------------------------
# Persist to DB
# ---------------------------------------------------------------------------

def ingest_markdown_file(
    file_path: Path,
    db_path: Path | None = None,
    doc_type: str | None = None,
    project: str | None = None,
    title: str | None = None,
    summary: str | None = None,
) -> tuple[str, list[str]]:
    """Ingest a markdown file into document + sections.

    Returns (doc_id, [section_ids]) — section_ids in ordinal order (leaves only;
    interior heading nodes that have no body content are not persisted as sections,
    but are represented by their first leaf child's path prefix).
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    text = file_path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_front_matter(text)

    # Meta overrides
    if not doc_type:
        doc_type = meta.get("doc_type") or meta.get("type") or "internal"
    if not project:
        project = meta.get("project") or None
    if not title:
        title = meta.get("title") or file_path.stem.replace("-", " ").replace("_", " ")
    if not summary:
        summary = meta.get("summary") or None

    # Compute source hash over full file content
    source_hash = hashlib.sha256(text.encode()).hexdigest()

    # Parse
    raw_sections = _parse_sections(body)
    intermediate = _build_intermediate(raw_sections)
    _compute_paths(raw_sections, intermediate)
    parent_map = _build_parent_map(raw_sections)
    raw_paths = _raw_section_paths(raw_sections)  # per-raw-section path (index-keyed)

    # Open DB
    from kb.config import load_config
    cfg = load_config()
    resolved_db = Path(db_path) if db_path else cfg.db_path
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    db_conn = DatabaseConnection(resolved_db, cfg.embedding_dim)
    conn = db_conn.conn
    init_schema(conn, cfg.embedding_dim)

    from kb.core.embedding import EmbeddingService
    _embedding = EmbeddingService(cfg.embedding_url, cfg.embedding_dim)
    docs_repo = DocumentsRepository(conn)
    secs_repo = DocumentSectionsRepository(conn, embedding_service=_embedding)

    # Create document record
    doc_id = docs_repo.add(
        title=title,
        doc_type=doc_type,
        source_path=str(file_path.resolve()),
        source_hash=source_hash,
        project=project,
        summary=summary,
    )

    # We need section_id per raw_section for parent linkage.
    # Since intermediate entries may be split across raw sections, we first
    # insert one "heading node" per raw section with a real heading, then
    # leaf sections under them.

    # Build heading-only nodes for interior sections (level >= 1 with heading)
    # Map raw_section_idx -> section_id in DB
    raw_section_db_ids: dict[int, str] = {}

    # One monotonic document-order ordinal across heading-nodes AND leaves
    # (was: heading=ri, leaf=entry["ordinal"] — two spaces that collided).
    _ord_map = _ordinal_map(
        [raw_paths[i] for i, rs in enumerate(raw_sections) if rs["level"] != 0]
        + [e["path"] for e in intermediate]
    )

    for ri, rs in enumerate(raw_sections):
        if rs["level"] == 0:
            continue  # preamble has no heading node
        parent_ri = parent_map.get(ri)
        parent_db_id = raw_section_db_ids.get(parent_ri) if parent_ri is not None else None

        # Path for this heading node = the raw section's own index-keyed path
        # (index-based; no text re-match — kb-86b074).
        heading_path = raw_paths[ri]

        # Insert a heading node (kind=prose, content=heading only if no leaf body)
        section_id, _ = secs_repo.upsert_by_path(
            document_id=doc_id,
            path=heading_path,
            content_hash=_content_hash(rs["heading"]),
            level=rs["level"],
            ordinal=_ord_map[heading_path],
            kind="prose",
            heading=rs["heading"],
            content=rs["heading"],
            embed_text=rs["heading"],
            parent_section_id=parent_db_id,
        )
        raw_section_db_ids[ri] = section_id

    # Now insert leaf sections (intermediate entries)
    section_ids: list[str] = []

    for entry in intermediate:
        # The originating raw section is carried on the entry (index-based;
        # no text re-match — kb-86b074).
        current_ri = entry["raw_idx"]

        parent_db_id: str | None = None
        if entry["level"] >= 1 and current_ri in raw_section_db_ids:
            parent_db_id = raw_section_db_ids[current_ri]
        elif entry["level"] >= 1:
            # parent from parent_map chain
            p_ri = parent_map.get(current_ri)
            if p_ri is not None:
                parent_db_id = raw_section_db_ids.get(p_ri)

        section_id, _ = secs_repo.upsert_by_path(
            document_id=doc_id,
            path=entry["path"],
            content_hash=_content_hash(entry["content"] or ""),
            level=entry["level"],
            ordinal=_ord_map[entry["path"]],
            kind=entry["kind"],
            heading=entry["heading"] or None,
            content=entry["content"],
            embed_text=entry.get("embed_text") or entry["content"],
            table_repr=entry.get("table_repr"),
            parent_section_id=parent_db_id,
        )
        section_ids.append(section_id)

    return doc_id, section_ids


# ---------------------------------------------------------------------------
# Section count helper (for `kb add -f` detection)
# ---------------------------------------------------------------------------

def count_heading_sections(text: str) -> int:
    """Count the number of ATX/setext heading-bounded sections in markdown text.

    Returns 0 if there are no headings (single-blob content).
    """
    _, body = _parse_front_matter(text)
    sections = _parse_sections(body)
    # Exclude level-0 preamble
    return sum(1 for s in sections if s["level"] > 0)


# ---------------------------------------------------------------------------
# CLI entry point (kb ingest md <file>)
# ---------------------------------------------------------------------------

def run(
    file_path: Path,
    db_path: Path | None = None,
    doc_type: str | None = None,
    project: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    dry_run: bool = False,
) -> int:
    """Ingest a markdown file. Returns 0 on success."""
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    if dry_run:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        _, body = _parse_front_matter(text)
        sections = _parse_sections(body)
        intermediate = _build_intermediate(sections)
        _compute_paths(sections, intermediate)
        n_headings = sum(1 for s in sections if s["level"] > 0)
        print(f"[dry-run] {file_path.name}: {n_headings} headings -> {len(intermediate)} leaves")
        for entry in intermediate:
            print(f"  [{entry['kind']:6s}] {entry['path']:12s} ord={entry['ordinal']} "
                  f"tokens={_token_count(entry['content'] or '')} heading={entry['heading']!r}")
        return 0

    try:
        doc_id, section_ids = ingest_markdown_file(
            file_path,
            db_path=db_path,
            doc_type=doc_type,
            project=project,
            title=title,
            summary=summary,
        )
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}", file=sys.stderr)
        return 1

    print(f"doc-id: {doc_id}")
    print(f"sections: {len(section_ids)}")
    for sid in section_ids:
        print(f"  {sid}")
    return 0
