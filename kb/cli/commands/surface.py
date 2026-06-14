"""CLI handler for `kb surface` — unified multi-source semantic surfacing.

Original --query mode composes three existing search methods:
  1. kb.search_python_symbols(query, limit, project)  -> code symbols
  2. kb.search(query, limit, project)                 -> findings (hybrid vector+FTS)
  3. kb._bridge.search(query, limit)                  -> bridge memory messages

New producer modes (kb-xob.1):
  --prompt   TEXT     what kb-prompt-surface would inject (SIM_FLOOR 0.42, top-3)
  --analysis TEXT     what kb-analysis-surface would inject (INTENT_RX + SIM_FLOOR 0.62)
  --file     PATH     what symbol_surface would inject on Read
  --issues   TEXT     what open_issues_surface would inject (vector+FTS over issues)
  --bridge   ID|-     what bridge inject would surface on a bridge message (by id or text)

All producer modes support --json for structured output.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Source queries for --query mode — each wrapped to return [] on error
# ---------------------------------------------------------------------------

def _query_symbols(kb: Any, query: str, limit: int, project: str | None, min_sim: float) -> list[dict[str, Any]]:
    try:
        raw = kb.search_python_symbols(query, limit=limit, project=project)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


def _query_findings(kb: Any, query: str, limit: int, project: str | None, min_sim: float) -> list[dict[str, Any]]:
    try:
        raw = kb.search(query, limit=limit, project=project)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


def _query_bridge(kb: Any, query: str, limit: int, min_sim: float) -> list[dict[str, Any]]:
    try:
        raw = kb._bridge.search(query, limit=limit)
        return [r for r in (raw or []) if r.get("similarity", 0) >= min_sim]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Formatting helpers for --query mode
# ---------------------------------------------------------------------------

def _fmt_symbol(r: dict[str, Any]) -> str:
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
    sim = r.get("similarity", 0.0)
    fid = r.get("id", "?")
    proj = r.get("project", "")
    summary = (r.get("summary") or r.get("content") or "")[:80]
    proj_tag = f" ({proj})" if proj else ""
    return f"[FIND  ~{sim:.2f}] {fid}{proj_tag}: {summary}"


def _fmt_bridge(r: dict[str, Any]) -> str:
    sim = r.get("similarity", 0.0)
    mid = r.get("id", "?")
    sender = r.get("sender", "?")
    subject = (r.get("subject") or r.get("body") or "")[:60]
    return f"[BRIDGE ~{sim:.2f}] #{mid} {sender}: {subject}"


# ---------------------------------------------------------------------------
# Producer mode helpers
# ---------------------------------------------------------------------------

def _run_producer_mode(kb: Any, args: Any, mode: str) -> None:
    """Dispatch to the appropriate produce_* function and render output."""
    from kb.surface.producers import (
        produce_prompt, produce_analysis, produce_symbols,
        produce_open_issues, produce_bridge,
    )

    as_json: bool = getattr(args, "json", False)
    project: str | None = getattr(args, "project", None)
    limit: int = getattr(args, "limit", 8)

    if mode == "prompt":
        text = args.prompt
        inj = produce_prompt(text, kb=kb, limit=limit)
    elif mode == "analysis":
        text = args.analysis
        inj = produce_analysis(text, kb=kb, limit=limit)
    elif mode == "file":
        inj = produce_symbols(file_path=args.file, project=project, kb=kb)
    elif mode == "issues":
        inj = produce_open_issues(args.issues, kb=kb, project=project)
    elif mode == "bridge":
        raw = args.bridge
        # Accept numeric id or raw text (- means read from args.bridge_text or just text)
        try:
            msg_id = int(raw)
            inj = produce_bridge(msg_id=msg_id, kb=kb)
        except (ValueError, TypeError):
            inj = produce_bridge(msg_text=raw, kb=kb)
    else:
        print(f"Unknown producer mode: {mode}")
        return

    if as_json:
        print(json.dumps({
            "producer": inj.producer,
            "fired": inj.fired,
            "context": inj.context,
            "hits": inj.hits,
        }, indent=2, default=str))
        return

    # Human-readable — name the producer AND the input it ran on
    inp = _input_preview(mode, args)
    if not inj.fired:
        print(f"[{mode}] did NOT fire on {inp} (no match above threshold)")
        return

    print(f"--- {mode} producer (on {inp}) ---")
    print(inj.context)


def _input_preview(mode: str, args: Any) -> str:
    """Short, single-line description of the input a producer ran on."""
    raw = {
        "prompt": getattr(args, "prompt", None),
        "analysis": getattr(args, "analysis", None),
        "file": getattr(args, "file", None),
        "issues": getattr(args, "issues", None),
        "bridge": getattr(args, "bridge", None),
        "all": getattr(args, "all_input", None),
    }.get(mode)
    if raw is None:
        return "(none)"
    if mode == "file":
        return raw  # a path — show it whole
    s = " ".join(str(raw).split())
    return f'"{s[:60]}…"' if len(s) > 60 else f'"{s}"'


def _run_all(kb: Any, args: Any) -> None:
    """Run ALL producers against one text input; show fired + did-NOT-fire."""
    from kb.surface.producers import (
        produce_prompt, produce_analysis, produce_symbols,
        produce_open_issues, produce_bridge,
    )
    text: str = args.all_input
    project: str | None = getattr(args, "project", None)
    limit: int = getattr(args, "limit", 8)

    injections = [
        produce_prompt(text, kb=kb, limit=limit),
        produce_analysis(text, kb=kb, limit=limit),
        produce_symbols(text=text, project=project, kb=kb),
        produce_open_issues(text, kb=kb, project=project),
        produce_bridge(msg_text=text, kb=kb),
    ]

    if getattr(args, "json", False):
        print(json.dumps({
            "input": {"mode": "all", "text": text},
            "injections": [
                {"producer": i.producer, "fired": i.fired,
                 "context": i.context, "hits": i.hits}
                for i in injections
            ],
        }, indent=2, default=str))
        return

    inp = _input_preview("all", args)
    fired = [i for i in injections if i.fired]
    quiet = [i.producer for i in injections if not i.fired]

    if not fired:
        print(f"No producers fired on {inp}.")
    for inj in fired:
        print(f"--- {inj.producer} producer (on {inp}) ---")
        print(inj.context)
        print()
    if quiet:
        print(f"producers that did NOT fire: {', '.join(quiet)}")


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def run_surface(kb: Any, args: Any) -> None:
    """Handle `kb surface`.

    Routes to producer mode if any of --prompt/--analysis/--file/--issues/--bridge
    are set. Falls back to legacy --query mode otherwise.
    """
    # Detect producer mode
    if getattr(args, "prompt", None) is not None:
        _run_producer_mode(kb, args, "prompt")
        return
    if getattr(args, "analysis", None) is not None:
        _run_producer_mode(kb, args, "analysis")
        return
    if getattr(args, "file", None) is not None:
        _run_producer_mode(kb, args, "file")
        return
    if getattr(args, "issues", None) is not None:
        _run_producer_mode(kb, args, "issues")
        return
    if getattr(args, "bridge", None) is not None:
        _run_producer_mode(kb, args, "bridge")
        return
    if getattr(args, "all_input", None) is not None:
        _run_all(kb, args)
        return

    # Legacy --query mode
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
