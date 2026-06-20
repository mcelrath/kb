"""CLI handler for `kb doc` — document navigation verbs.

Subcommands:
  list   - list document roots from the documents table
  toc    - print the heading tree (section ids + paths) for a document
  get    - fetch a section by path; --subtree includes descendants
"""

from __future__ import annotations

from typing import Any

import kb.cli.output as output


def _indent(level: int) -> str:
    return "  " * (level - 1)


def run_doc(kb: Any, args: Any, doc_parser: Any) -> None:
    """Dispatch `kb doc` subcommands."""
    doc_cmd = getattr(args, "doc_cmd", None)

    if doc_cmd == "list":
        _run_list(kb, args)
    elif doc_cmd == "toc":
        _run_toc(kb, args)
    elif doc_cmd == "get":
        _run_get(kb, args)
    else:
        doc_parser.print_help()


def _run_list(kb: Any, args: Any) -> None:
    """List document roots."""
    from kb.entities.documents import DocumentsRepository

    docs_repo = DocumentsRepository(kb.conn)
    project: str | None = getattr(args, "project", None)
    doc_type: str | None = getattr(args, "type", None)
    as_json: bool = getattr(args, "json", False)

    docs = docs_repo.list(project=project, doc_type=doc_type)

    if as_json:
        import json
        print(json.dumps(docs, indent=2, default=str))
        return

    if not docs:
        print("No documents found.")
        return

    for d in docs:
        proj_tag = f" [{d['project']}]" if d.get("project") else ""
        type_tag = f" ({d['doc_type']})" if d.get("doc_type") else ""
        summary = f" — {d['summary'][:60]}" if d.get("summary") else ""
        doc_id = output.c(str(d["id"]), "cyan")
        title = output.c(str(d["title"]), "bold")
        print(output.fit_line(f"{doc_id}{proj_tag}{type_tag}  {title}{summary}"))


def _run_toc(kb: Any, args: Any) -> None:
    """Print the heading tree for a document."""
    from kb.entities.document_sections import DocumentSectionsRepository

    doc_id: str = args.doc_id
    as_json: bool = getattr(args, "json", False)

    sections_repo = DocumentSectionsRepository(kb.conn)
    sections = sections_repo.list_by_document(doc_id)

    if not sections:
        print(f"No sections found for document {doc_id!r}.")
        return

    if as_json:
        import json
        rows = [
            {
                "id": s["id"],
                "level": s["level"],
                "path": s["path"],
                "heading": s["heading"],
            }
            for s in sections
        ]
        print(json.dumps(rows, indent=2, default=str))
        return

    for s in sections:
        level = s["level"] or 1
        heading = s["heading"] or "(no heading)"
        path = s["path"] or ""
        sid = s["id"]
        indent = _indent(level)
        hashes = output.c("#" * level, "blue")
        heading_str = output.c(heading, "bold")
        sid_str = output.c(f"[{sid}]", "dim")
        print(output.fit_line(f"{indent}{hashes} {heading_str}  {sid_str}  path={path}"))


def _run_get(kb: Any, args: Any) -> None:
    """Fetch a section by path; --subtree includes descendants."""
    from kb.entities.document_sections import DocumentSectionsRepository

    doc_id: str = args.doc_id
    path: str = args.path
    subtree: bool = getattr(args, "subtree", False)
    as_json: bool = getattr(args, "json", False)

    sections_repo = DocumentSectionsRepository(kb.conn)

    # Find the section at the given path
    row = kb.conn.execute(
        """SELECT * FROM document_sections
           WHERE document_id = ? AND path = ? AND status = 'active'""",
        (doc_id, path),
    ).fetchone()

    if row is None:
        print(f"No active section found at path {path!r} in document {doc_id!r}.")
        return

    target = dict(row)

    if not subtree:
        results = [target]
    else:
        # Include all descendants: collect sections whose path starts with target path
        # (for numeric paths like "1.2", match "1.2" and "1.2.*")
        all_sections = sections_repo.list_by_document(doc_id)
        prefix = path + "."
        results = [
            s for s in all_sections
            if s["path"] == path or s["path"].startswith(prefix)
        ]

    if as_json:
        import json
        print(json.dumps(results, indent=2, default=str))
        return

    for s in results:
        level = s["level"] or 1
        heading = s["heading"] or "(no heading)"
        sid = s["id"]
        kind = s["kind"]
        indent = _indent(level)
        hashes = output.c("#" * level, "blue")
        heading_str = output.c(heading, "bold")
        sid_str = output.c(f"[{sid}]", "dim")
        # heading line: single-line, fit to width; body content below is multi-line, not truncated
        print(f"{indent}{hashes} {heading_str}  {sid_str}  kind={kind}  path={s['path']}")
        if s.get("asset_path"):
            print(f"  asset_path: " + str(s.get("asset_path")))
        if s.get("content"):
            # Print a short excerpt
            content = str(s["content"])
            excerpt = content[:200].replace("\n", " ")
            if len(content) > 200:
                excerpt += "…"
            print(f"  {excerpt}")
        print()
