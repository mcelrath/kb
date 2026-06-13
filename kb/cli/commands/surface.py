"""CLI handler for `kb surface` — unified multi-source semantic surfacing.

Composes three existing search methods into a single function-first view:
  1. kb.search_python_symbols(query, limit, project)  -> code symbols
  2. kb.search(query, limit, project)                 -> findings (hybrid vector+FTS)
  3. kb._bridge.search(query, limit)                  -> bridge memory messages

ORDER: code symbols first (most actionable for "am I about to reimplement this"),
then findings, then bridge memory.  Each source is queried independently; a source
that errors or returns nothing is skipped gracefully.

Agent hooks (kb-prompt-surface, symbol_surface, kb-analysis-surface) can migrate
to call `kb surface --json` instead of their own queries — that rewiring is a
SEPARATE follow-up; this module only adds the command itself.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Source queries — each wrapped to return [] on any error (embed-down tolerant)
# ---------------------------------------------------------------------------

def _query_symbols(kb: Any, query: str, limit: int, project: str | None, min_sim: float) -> list[dict[str, Any]]:
    """Search python_symbols_vec; returns [] if table empty or embed down."""
    try:
        raw = kb.search_python_symbols(query, limit=limit, project=project)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


def _query_findings(kb: Any, query: str, limit: int, project: str | None, min_sim: float) -> list[dict[str, Any]]:
    """Hybrid vector+FTS findings search; returns [] on any error."""
    try:
        raw = kb.search(query, limit=limit, project=project)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


def _query_bridge(kb: Any, query: str, limit: int, min_sim: float) -> list[dict[str, Any]]:
    """Bridge message search; returns [] if table empty, embed down, or any error."""
    try:
        raw = kb._bridge.search(query, limit=limit)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_symbol(r: dict[str, Any]) -> str:
    """One-line [CODE] entry."""
    sim = r.get("similarity", 0.0)
    name = r.get("name", "?")
    module = r.get("module", "")
    kind = r.get("kind", "?")
    fpath = r.get("file", "")
    line = r.get("line", "")
    qualified = f"{module}.{name}" if module else name
    location = f"{fpath}:{line}" if fpath and line else (fpath or "")
    return f"[CODE  ~{sim:.2f}] {qualified}  ({kind})  {location}"


def _fmt_finding(r: dict[str, Any]) -> str:
    """One-line [FIND] entry."""
    sim = r.get("similarity", 0.0)
    fid = r.get("id", "?")
    proj = r.get("project", "")
    summary = (r.get("summary") or r.get("content") or "")[:80]
    proj_tag = f" ({proj})" if proj else ""
    return f"[FIND  ~{sim:.2f}] {fid}{proj_tag}: {summary}"


def _fmt_bridge(r: dict[str, Any]) -> str:
    """One-line [BRIDGE] entry."""
    sim = r.get("similarity", 0.0)
    mid = r.get("id", "?")
    sender = r.get("sender", "?")
    subject = (r.get("subject") or r.get("body") or "")[:60]
    return f"[BRIDGE ~{sim:.2f}] #{mid} {sender}: {subject}"


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def run_surface(kb: Any, args: Any) -> None:
    """Handle `kb surface <query>`.

    Queries the selected sources (default: all three), merges results in
    function-first order, and prints a compact tagged list.  --json emits a
    structured object {symbols:[...], findings:[...], bridge:[...]} for
    hook/script consumption.
    """
    query: str = args.query
    n: int = args.limit
    project: str | None = getattr(args, "project", None)
    min_sim: float = getattr(args, "min_sim", 0.45)
    sources_raw: str = getattr(args, "sources", "code,findings,bridge")
    sources = {s.strip().lower() for s in sources_raw.split(",")}
    as_json: bool = getattr(args, "json", False)

    symbols: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []

    if "code" in sources:
        symbols = _query_symbols(kb, query, n, project, min_sim)

    if "findings" in sources:
        findings = _query_findings(kb, query, n, project, min_sim)

    if "bridge" in sources:
        bridge = _query_bridge(kb, query, n, min_sim)

    if as_json:
        print(json.dumps({"symbols": symbols, "findings": findings, "bridge": bridge},
                         indent=2, default=str))
        return

    total = len(symbols) + len(findings) + len(bridge)
    if total == 0:
        print("No results found")
        return

    if symbols:
        for r in symbols:
            print(_fmt_symbol(r))

    if findings:
        for r in findings:
            print(_fmt_finding(r))

    if bridge:
        for r in bridge:
            print(_fmt_bridge(r))
