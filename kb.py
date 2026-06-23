#!/usr/bin/env python3
"""
Knowledge Base CLI - Command-line interface for the KB system.

This module provides the CLI for interacting with the Knowledge Base,
including web server functionality. The core library is in the kb/ package.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from kb.markdown import format_finding_markdown  # noqa: F401 (re-exported for CLI)
from kb.cli import output as output


def _is_agent_context() -> bool:
    """Detect if called from an agent (non-interactive) context.

    Checks in order:
      KB_AGENT=1/0  — explicit override
      CLAUDECODE=1  — set by Claude Code in all subprocesses
      stdout isatty — fallback: pipe/subprocess = agent
    """
    return output.is_agent()


# Re-export so call-sites using `kb.AGENT_MODE` continue to work.
AGENT_MODE = output.AGENT_MODE

# Import from kb package
from kb import (
    KnowledgeBase,
    FINDING_TYPES,
    NOTATION_DOMAINS,
    DEFAULT_DB_PATH,
)

# Optional: rich for terminal markdown rendering
try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Optional: starlette/uvicorn for web server
try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, StreamingResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket
    import asyncio
    import uvicorn
    SERVE_AVAILABLE = True
except ImportError:
    SERVE_AVAILABLE = False

BRIDGE_MESSAGES_PATH = Path.home() / ".agent-bridge" / "messages.jsonl"
BRIDGE_AGENTS_PATH = Path.home() / ".agent-bridge" / "agents.json"


def _bridge_msg_for_recipient(msg: dict, recipient: str) -> bool:
    """Return True if msg is addressed to recipient or is a broadcast to 'all'."""
    to = msg.get("to", [])
    if isinstance(to, str):
        to = [t.strip() for t in to.split(",")]
    if not isinstance(to, list):
        to = [str(to)]
    # Include broadcast ('all') and direct addressing
    return recipient in to or "all" in to


def _parse_bridge_messages(recipient: str | None, limit: int, last_event_id: int | None = None) -> list[dict]:
    """Read messages.jsonl, filter by recipient (or return all if None), return newest-last.

    If last_event_id is given, only return messages with numeric id > last_event_id.
    """
    if not BRIDGE_MESSAGES_PATH.exists():
        return []
    msgs = []
    try:
        with open(BRIDGE_MESSAGES_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if last_event_id is not None:
                    msg_id = msg.get("id")
                    if msg_id is not None:
                        try:
                            if int(msg_id) <= int(last_event_id):
                                continue
                        except (TypeError, ValueError):
                            pass
                if recipient is None or _bridge_msg_for_recipient(msg, recipient):
                    msgs.append(msg)
    except OSError:
        return []
    # Return last `limit` messages (newest-last = natural order, slice tail)
    return msgs[-limit:] if limit > 0 else msgs


PENDING_QUEUE_DIR = Path.home() / ".claude" / "pending-kb-adds"
KB_STATE_DIR = "/tmp/claude-kb-state"


def _load_session_seen_ids() -> set[str]:
    """Return kb IDs already in agent context this session.

    Walks the PPID chain to find the Claude Code session file written by
    history-isolation.sh, then reads ${session_id}-kb-seen (maintained by
    dedupe-kb-get.sh and kb-search-track.sh).  The file contains one kb-ID
    per line: written on kb add, kb get, and kb search result output.
    """
    try:
        pid = os.getpid()
        for _ in range(6):
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            pid = int(line.split()[1])
                            break
                    else:
                        break
            except OSError:
                break
            session_file = f"{KB_STATE_DIR}/session-{pid}"
            if os.path.exists(session_file):
                with open(session_file) as f:
                    session_id = f.read().strip()
                seen_file = f"{KB_STATE_DIR}/{session_id}-kb-seen"
                if os.path.exists(seen_file):
                    with open(seen_file) as f:
                        return {ln.strip() for ln in f if ln.strip().startswith("kb-")}
                return set()
    except Exception:
        pass
    return set()


def _queue_async_add(
    content: str,
    finding_type: str | None,
    project: str | None,
    sprint: str | None,
    tags: list[str] | None,
    evidence: str | None,
    summary: str | None = None,
) -> None:
    """Write a queue file readable by `kb flush-pending` and detach a flusher.

    Returns immediately; the flusher runs in its own session so the parent shell
    (and the agent) does not wait on embedding/LLM I/O.
    """
    import os
    import subprocess
    import secrets
    from datetime import datetime, timezone

    PENDING_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(3)
    qfile = PENDING_QUEUE_DIR / f"{stamp}-{rand}.txt"

    header_lines = []
    if finding_type:
        header_lines.append(f"# type: {finding_type}")
    if project:
        header_lines.append(f"# project: {project}")
    if sprint:
        header_lines.append(f"# sprint: {sprint}")
    if tags:
        header_lines.append(f"# tags: {','.join(tags)}")
    if evidence:
        # Header is single-line; multi-line evidence belongs in content.
        header_lines.append(f"# evidence: {evidence}")
    if summary:
        header_lines.append(f"# summary: {summary}")
    payload = "\n".join(header_lines) + ("\n\n" if header_lines else "") + content + "\n"

    tmp = qfile.with_suffix(".txt.partial")
    tmp.write_text(payload)
    tmp.rename(qfile)  # atomic publish

    # Detach a flusher. Same interpreter, same script. The flusher probes the
    # embedding server itself and exits silently if it's down.
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "flush-pending", "--quiet"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    print(f"Queued: {qfile.name}")


def parse_markdown_findings(file_path: Path) -> list[dict]:
    """Parse a markdown file and extract findings.

    Looks for patterns like:
    - **[SUCCESS]** or **[FAILURE]** markers
    - Bullet points with key findings
    - Sections with results/conclusions
    """
    import re

    content = file_path.read_text()
    findings = []

    # Pattern 1: Explicit markers like **[SUCCESS]**, **[FAILURE]**, etc.
    marker_pattern = re.compile(
        r'\*\*\[(SUCCESS|FAILURE|EXPERIMENT|DISCOVERY)\]\*\*[:\s]*(.+?)(?=\n\n|\n\*\*\[|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    for match in marker_pattern.finditer(content):
        finding_type = match.group(1).lower()
        text = match.group(2).strip()
        findings.append({
            'type': finding_type,
            'content': text[:500],
            'evidence': None,
        })

    # Pattern 2: Key result sections (## Results, ## Findings, ## Conclusions)
    section_pattern = re.compile(
        r'^##\s+(Results?|Findings?|Conclusions?|Key\s+Findings?)\s*\n(.*?)(?=\n##|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    for match in section_pattern.finditer(content):
        section_content = match.group(2).strip()
        # Extract bullet points
        bullets = re.findall(r'^[-*]\s+(.+)$', section_content, re.MULTILINE)
        for bullet in bullets:
            if len(bullet) > 30:  # Skip short bullets
                findings.append({
                    'type': 'discovery',
                    'content': bullet.strip()[:500],
                    'evidence': None,
                })

    # Pattern 3: Numbered conclusions/results
    numbered_pattern = re.compile(r'^\d+\.\s+\*\*(.+?)\*\*[:\s]*(.+?)(?=\n\d+\.|\n\n|\Z)', re.MULTILINE | re.DOTALL)
    for match in numbered_pattern.finditer(content):
        title = match.group(1).strip()
        desc = match.group(2).strip()
        full = f"{title}: {desc}" if desc else title
        if len(full) > 40:
            findings.append({
                'type': 'discovery',
                'content': full[:500],
                'evidence': None,
            })

    # Deduplicate by content similarity
    seen = set()
    unique = []
    for f in findings:
        key = f['content'][:100].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def parse_script_findings(file_path: Path) -> list[dict]:
    """Parse a Python script and extract docstrings as findings.

    Extracts:
    - Module-level docstrings
    - Class docstrings with class name
    - Function/method docstrings with function name
    """
    import ast

    content = file_path.read_text()
    findings = []

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [{"type": "failure", "content": f"Syntax error in {file_path}: {e}", "evidence": None}]

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) > 30:
        findings.append({
            "type": "discovery",
            "content": f"[{file_path.name}] {module_doc[:500]}",
            "evidence": None,
        })

    # Class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[class {node.name}] {doc[:500]}",
                    "evidence": None,
                })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[{node.name}()] {doc[:500]}",
                    "evidence": None,
                })

    return findings


_TYPE_COLORS = {
    "success":    "green",
    "failure":    "red",
    "experiment": "yellow",
    "discovery":  "cyan",
    "correction": "magenta",
}
_TYPE_ABBREV = {
    "success":    "SUC",
    "failure":    "FAI",
    "experiment": "EXP",
    "discovery":  "DIS",
    "correction": "COR",
}


_TYPE_ABBREV_AGENT = {
    "correction": "COR", "discovery": "DIS", "success": "SUC",
    "failure": "FAI", "experiment": "EXP",
}
# Tags that add no information in the one-liner (they duplicate the type field)
_SKIP_TAGS = frozenset({"discovery", "success", "failure", "experiment", "correction",
                        "core-result", "technique", "detail", "proven", "heuristic",
                        "open-problem"})


def _fmt_age(created_at: str | None) -> str:
    """Human-readable age: 3d, 2w, 4m, 1y."""
    if not created_at:
        return ""
    try:
        from datetime import datetime
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        days = (datetime.now(created.tzinfo) - created).days
        if days < 1:   return "today"
        if days < 14:  return f"{days}d"
        if days < 60:  return f"{days // 7}w"
        if days < 365: return f"{days // 30}m"
        return f"{days // 365}y"
    except Exception:
        return ""


def _fmt_one_line(finding: dict) -> str:
    """One-line summary for search/list/related. Colored in user mode, plain in agent mode."""
    # Document-section hits (from ingested PDFs/markdown) have no finding `type`
    # (they carry result_type='section' + kind); render a DOC tag + path instead
    # of falling through to '???'.
    RESET = output.RESET
    if finding.get("result_type") == "section":
        kind = finding.get("kind", "prose")
        tag = {"table": "DOC·tbl", "figure": "DOC·fig"}.get(kind, "DOC")
        sim = finding.get("similarity")
        path = finding.get("path", "")
        text = (finding.get("heading")
                or (finding.get("content") or "").split("\n")[0][:100])
        proj = finding.get("project") or "?"
        if output.AGENT_MODE:
            sim_str = f" ({sim:.2f})" if sim is not None else ""
            return f"{finding['id']}{sim_str} [{tag} {path}] ({proj})  {text}"
        sim_str = (f" {output.c(f'({sim:.2f})', output.sim_color(sim))}" if sim is not None else "")
        return (f"{output.c(f'[{tag} {path}]', 'cyan')} {output.c(finding['id'], 'dim')}"
                f"{sim_str} {output.c(f'({proj})', 'dim')}  {text}")

    text = finding.get("summary") or finding["content"].split("\n")[0][:100]
    sim = finding.get("similarity")

    if output.AGENT_MODE:
        sim_str = f" ({sim:.2f})" if sim is not None else ""
        type_abbr = _TYPE_ABBREV_AGENT.get(finding.get("type", ""), "???")
        age = _fmt_age(finding.get("created_at"))
        tags = [t for t in (finding.get("tags") or []) if t not in _SKIP_TAGS]
        meta_parts = [type_abbr]
        if age:
            meta_parts.append(age)
        if tags:
            meta_parts.append(",".join(tags[:3]))
        meta = "[" + " ".join(meta_parts) + "]"
        return f"{finding['id']}{sim_str} {meta}  {text}"

    color = _TYPE_COLORS.get(finding.get("type", ""), "")
    abbr  = _TYPE_ABBREV.get(finding.get("type", ""), "???")
    sim_str = (f" {output.c(f'({sim:.2f})', output.sim_color(sim))}" if sim is not None else "")
    proj_str = f"({finding['project']})" if finding.get("project") else ""
    proj = f" {output.c(proj_str, 'dim')}" if proj_str else ""
    return f"{output.c(f'[{abbr}]', color)} {output.c(finding['id'], 'dim')}{sim_str}{proj}  {text}"


def format_results(findings: list[dict]) -> str:
    """Render the one-line search/list view for a whole result set.

    Agent mode: one `_fmt_one_line` per row, joined — byte-identical to the prior
    per-row output, so agents/hooks that parse it are unaffected.
    User mode: column-aligned, colored table (tag / id / similarity / project /
    text) so a human sees tidy columns instead of ragged per-row strings. Handles
    both finding rows and ingested document-section rows (which carry a DOC tag).
    """
    if not findings:
        return ""
    if output.AGENT_MODE:
        return "\n".join(_fmt_one_line(f) for f in findings)

    RESET = output.RESET

    rows = []  # (tag_raw, tag_color, fid, sim, sim_raw, proj_raw, text)
    for f in findings:
        sim = f.get("similarity")
        sim_raw = f"({sim:.2f})" if isinstance(sim, (int, float)) else ""
        if f.get("result_type") == "section":
            kind = f.get("kind", "prose")
            tg = {"table": "DOC·tbl", "figure": "DOC·fig"}.get(kind, "DOC")
            tag_raw = f"[{tg} {f.get('path', '')}]"
            tag_color = "cyan"
            proj_raw = f"({f.get('project') or '?'})"
            text = f.get("heading") or (f.get("content") or "").split("\n")[0][:100]
        else:
            tag_raw = f"[{_TYPE_ABBREV.get(f.get('type', ''), '???')}]"
            tag_color = _TYPE_COLORS.get(f.get("type", ""), "")
            proj_raw = f"({f['project']})" if f.get("project") else ""
            text = f.get("summary") or f["content"].split("\n")[0][:100]
        rows.append((tag_raw, tag_color, f["id"], sim, sim_raw, proj_raw, text))

    tagw = max(len(r[0]) for r in rows)
    idw = max(len(r[2]) for r in rows)
    simw = max(len(r[4]) for r in rows)
    projw = max(len(r[5]) for r in rows)

    out = []
    for tag_raw, tag_color, fid, sim, sim_raw, proj_raw, text in rows:
        cells = [output.c(f"{tag_raw:<{tagw}}", tag_color),
                 output.c(f"{fid:<{idw}}", "dim")]
        if simw:
            cells.append(output.c(f"{sim_raw:<{simw}}", output.sim_color(sim))
                         if sim is not None else " " * simw)
        if projw:
            cells.append(output.c(f"{proj_raw:<{projw}}", "dim") if proj_raw else " " * projw)
        row = " ".join(cells) + f"  {text}"
        out.append(output.fit_line(row))
    return "\n".join(out)


def format_finding(finding: dict, verbose: bool = False) -> str:
    """Format a finding for terminal display (list/search output)."""
    RESET = output.RESET
    type_color = _TYPE_COLORS.get(finding["type"], "")
    header = f"[{output.c(finding['type'].upper(), type_color)}] {output.c(finding['id'], 'dim')}"

    if finding.get("project"):
        proj_label = f"({finding['project']})"
        header += f" {output.c(proj_label, 'dim')}"

    if finding.get("similarity") is not None:
        sim = finding["similarity"]
        # Thresholds differ slightly here (0.8/0.6) vs sim_color (0.7/0.5) — preserve original
        sc = "green" if sim >= 0.8 else "yellow" if sim >= 0.6 else "red"
        header += f" {output.c(f'({sim:.2f})', sc)}"

    lines = [header, f"  {finding['content']}"]

    if verbose:
        if finding.get("evidence"):
            ev = finding["evidence"]
            ev_text = f"{ev[:200]}..." if len(ev) > 200 else ev
            lines.append("  " + output.c(f"Evidence: {ev_text}", "dim"))
        if finding.get("supersedes_id"):
            lines.append("  " + output.c(f"Supersedes: {finding['supersedes_id']}", "dim"))
        if finding.get("tags"):
            lines.append("  " + output.c(f"Tags: {', '.join(finding['tags'])}", "dim"))
        lines.append("  " + output.c(f"Created: {finding['created_at']}", "dim"))
        if finding.get("similarity"):
            lines.append("  " + output.c(f"Similarity: {finding['similarity']:.3f}", "dim"))

    return "\n".join(lines)


def format_finding_summary(finding: dict) -> str:
    """Format a finding as a single-line summary."""
    type_abbrev = {
        "success": "✓",
        "failure": "✗",
        "experiment": "?",
        "discovery": "→",
        "correction": "↻",
    }
    symbol = type_abbrev.get(finding["type"], "·")
    content = finding["content"][:80] + "..." if len(finding["content"]) > 80 else finding["content"]
    return f"{symbol} {finding['id']}: {content}"


def _run_refresh(kb, rows, dry_run: bool, commit_every: int = 1, label: str = "refresh"):
    """Core loop: summarize + retag + reembed each finding row.

    Embedding (ash:8081) and LLM (tardis:9510) run concurrently per row.
    All I/O completes before any DB write opens. Each row is written with its
    own BEGIN IMMEDIATE/COMMIT via kb.update_finding_refresh() — lock held for
    microseconds, not seconds.

    rows: list of (id, project, content, evidence)
    Returns (ok, fail) counts.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from kb.validation import serialize_f32, l2_normalize

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    ok = fail = 0
    total = len(rows)
    t0 = _time.time()

    bar = _tqdm(total=total, desc=label, unit="row",
                dynamic_ncols=True) if _tqdm and not dry_run else None
    if bar:
        bar.set_postfix(ok=0, fail=0)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for fid, fproject, content, evidence in rows:
                embed_text = content + (" " + evidence if evidence else "")

                if dry_run:
                    summary = kb._analyzer.generate_summary(content, evidence)
                    print(f"[DRY] {fid} ({fproject}): {summary}")
                    ok += 1
                    if bar:
                        bar.set_postfix(ok=ok, fail=fail)
                        bar.update(1)
                    continue

                existing_tags = kb._fetch_existing_tags(fproject)

                def _embed(t=embed_text):
                    return kb._embedding._embed_remote(t, max_retries=5, base_delay=1.5)

                def _llm(c=content, e=evidence, et=existing_tags):
                    s = kb._analyzer.generate_summary(c, e)
                    t = kb._analyzer.suggest_tags(c, et)
                    return s, t

                # Both network calls run concurrently; no DB lock held.
                embed_fut = pool.submit(_embed)
                llm_fut   = pool.submit(_llm)

                embedding = None
                try:
                    embedding = serialize_f32(l2_normalize(embed_fut.result()))
                except Exception as e:
                    (bar.write if bar else print)(f"  EMBED FAIL {fid}: {e}")

                summary = tags = None
                try:
                    summary, tags = llm_fut.result()
                except Exception as e:
                    (bar.write if bar else print)(f"  LLM FAIL {fid}: {e}")

                if summary and len(summary) >= 10:
                    kb.update_finding_refresh(fid, summary, tags, embedding)
                    ok += 1
                else:
                    fail += 1
                    (bar.write if bar else print)(
                        f"  FAIL {fid} ({fproject}): {(content or '')[:60]!r}"
                    )

                if bar:
                    bar.set_postfix(ok=ok, fail=fail)
                    bar.update(1)

    except KeyboardInterrupt:
        if bar:
            bar.write(f"\nInterrupted at {ok+fail}/{total}")
            bar.close()
        else:
            print(f"\nInterrupted at {ok+fail}/{total}")
        elapsed = _time.time() - t0
        print(f"{label}: ok={ok} fail={fail} (interrupted after {elapsed/60:.1f}m)")
        return ok, fail

    if bar:
        bar.close()
    elapsed = _time.time() - t0
    print(f"{label}: ok={ok} fail={fail} total={total} elapsed={elapsed/60:.1f}m")
    return ok, fail


def _fetch_refresh_rows(kb, ids=None, project=None, all_rows=False, limit=0):
    """Build the findings row list for refresh/retag/resummarize."""
    return kb.fetch_refresh_rows(ids=ids, project=project, all_rows=all_rows, limit=limit)


def _backfill_statement_pure(kb, project=None, limit=None, workers=8, dry_run=False):
    """Backfill statement_pure for lean theorems using the KB's LLM client."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = kb._theorems.fetch_missing_statement_pure(project=project, limit=limit)
    print(f"  theorem backfill: {len(rows)} without statement_pure")
    if not rows:
        return {"updated": 0, "failed": 0}

    if dry_run:
        for tid, lean_name, stmt in rows[:3]:
            print(f"  [DRY] {lean_name}: {stmt[:80]}")
        return {"updated": 0, "failed": 0}

    PROMPT = (
        "Restate this Lean 4 theorem in pure mathematical language. "
        "No Lean syntax, no type annotations. Standard math notation. "
        "One sentence, under 30 words.\n\nLean:\n{statement}\n\nMath:"
    )

    def restate_one(row):
        tid, lean_name, statement = row
        result = kb._analyzer.llm_client.complete(
            PROMPT.format(statement=statement[:600]),
            max_tokens=80, temperature=0.1, timeout=30,
        )
        if result:
            result = result.strip().strip('"').strip("'")
        return tid, lean_name, result or None

    updated = failed = 0
    t0 = _time.time()
    conn = kb._theorems.conn

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(restate_one, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            tid, lean_name, pure = fut.result()
            if pure:
                kb._theorems.set_statement_pure(tid, pure)
                updated += 1
            else:
                failed += 1
            if i % 100 == 0:
                conn.commit()
                elapsed = _time.time() - t0
                rate = i / max(elapsed, 0.001)
                print(f"  {i}/{len(rows)} done  {rate:.1f}/s", flush=True)

    conn.commit()

    # Re-embed updated theorems
    if updated > 0:
        print(f"  re-embedding {updated} theorems...")
        updated_rows = conn.execute(
            "SELECT id, statement_pure FROM lean_theorems "
            "WHERE statement_pure IS NOT NULL AND statement_pure != ''"
        ).fetchall()
        for j, (tid, pure) in enumerate(updated_rows):
            kb._theorems.reembed_statement_pure(tid, pure)
            if j % 100 == 0:
                conn.commit()
        conn.commit()

    elapsed = _time.time() - t0
    print(f"  theorem backfill done: updated={updated} failed={failed} elapsed={elapsed:.0f}s")
    return {"updated": updated, "failed": failed}


_AGENT_CMDS = [
    ("add",           '-t TYPE -p PROJECT "content"   record a finding (sync, prints kb-id)'),
    ("search",        '"query" [-n N] [-p PROJECT] [-t TYPE] [-l]   semantic search'),
    ("list",          '[-n N] [-p PROJECT] [-s SPRINT] [-t TYPE] [-l]   list findings'),
    ("get",           "<kb-id>   full entry"),
    ("correct",       '<kb-id> "new content" [-r reason]   supersede a finding'),
    ("related",   "<kb-id> [-n N]   find semantically similar findings"),
    ("surface",   '"query" | --prompt/--analysis/--file/--all   multi-source surface (code symbols + findings + bridge)'),
]

_MAINT_CMDS = [
    ("refresh",       "retag + resummarize + reembed  [-p PROJECT] [--all] [--theorems]"),
    ("review",        "findings needing attention  [-p PROJECT]"),
    ("questions",     "LLM: identify research gaps  [-p PROJECT] [-n N] [-i N] [query]"),
    ("ask",           'LLM: answer a question from KB  "question" [-p PROJECT]'),
    ("stats",         "counts by type and project"),
    ("flush-pending", "drain the offline-add queue"),
    ("ingest",        "md | pdf | lean | python | typescript | rust | tex | scripts | personas   <file|dir>"),
    ("delete",        "<kb-id> [--force]"),
    ("export",        "<file.json> [-p PROJECT]"),
    ("import",        "<file.json>"),
    ("serve",         "[--port 8000]   HTTP/SSE server (bridge watch/messages/agents + kb endpoints)"),
    ("configure",     "embedding + LLM endpoints (health-checked) + summary mode; [--llm-url][--install-server][--project TAG --enable-tracker]"),
    ("embed-status",  "show configured-vs-stored embedding model/dim + verdict"),
    ("reembed",       "re-embed all findings  [--force]  (required after an embedding model/dim change)"),
]

def _print_main_help():
    W = 14  # column width for command names
    if AGENT_MODE:
        print("kb <command>   (agent mode)\n")
        for cmd, desc in _AGENT_CMDS:
            print(f"  {cmd:<{W}}{desc}")
        print()
        for cmd, desc in _MAINT_CMDS:
            print(f"  {cmd:<{W}}{desc}")
        print("\nIssue tracking: use the  kbt  command (kb-native tracker, bd-compatible:")
        print("  kbt ready | list | create | show | update | close | dep | blocked).")
        print("  Defaults to the kb-native tracker; a non-empty legacy .beads warns to migrate")
        print("  (kbt bead-migrate). Explicit .beads backend: / .kbt marker / KBT_BACKEND override.")
        print("kb-server (bridge + kb/issue HTTP endpoints): install the systemd --user unit with")
        print("  kb configure --install-server [--server-port 8765].")
        print("\nRun any command with --help for full flag list.")
        print("Set KB_AGENT=0 for user mode.")
    else:
        bold   = "\033[1m"
        cyan   = "\033[36m"
        yellow = "\033[33m"
        dim    = "\033[2m"
        reset  = "\033[0m"

        print(f"{bold}Knowledge Base{reset}  {dim}(set CLAUDECODE=1 or KB_AGENT=1 for agent mode){reset}\n")

        print(f"{bold}Agent commands:{reset}")
        for cmd, desc in _AGENT_CMDS:
            parts = desc.split("   ")
            summary = parts[-1] if len(parts) > 1 else desc
            print(f"  {cyan}{cmd:<{W}}{reset}{summary}")
        print()

        print(f"{bold}Maintenance:{reset}")
        for cmd, desc in _MAINT_CMDS:
            # desc format: "plain description  [-flags]" or just "[-flags]"
            # Split on first double-space to separate description from flags
            if "  " in desc:
                plain, flags = desc.split("  ", 1)
                flags = "  " + flags
            else:
                plain, flags = desc, ""
            print(f"  {yellow}{cmd:<{W}}{reset}{plain}{dim}{flags}{reset}")
        print()

        print(f"{bold}Issue tracking:{reset}  {cyan}kbt{reset}{dim}  (kb-native, bd-compatible: ready|list|create|show|update|close|dep|blocked){reset}")
        print(f"  {dim}defaults to the kb-native tracker; a non-empty legacy .beads warns to run kbt bead-migrate{reset}")
        print(f"{bold}Server:{reset}  {dim}kb configure --install-server  installs the kb-server systemd --user unit{reset}")
        print()

        print(f"{bold}Options:{reset}")
        print(f"  {dim}{'--db PATH':<{W}}database path (default: ~/.cache/kb/knowledge.db){reset}")
        print(f"  {dim}{'-h, --help':<{W}}show this help{reset}")


def main():
    from kb.cli.commands import findings as _cmd_findings
    from kb.cli.commands import admin as _cmd_admin
    from kb.cli.commands import maintenance as _cmd_maintenance
    from kb.cli.commands import ingest as _cmd_ingest
    from kb.cli.commands import bridge as _cmd_bridge
    from kb.cli.commands import lean as _cmd_lean
    from kb.cli.commands import serve as _cmd_serve
    from kb.cli.commands import misc as _cmd_misc
    from kb.cli.commands import surface as _cmd_surface
    from kb.cli.commands import doc as _cmd_doc

    parser = argparse.ArgumentParser(
        description="Knowledge Base",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", default=False)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Database path")
    subparsers = parser.add_subparsers(dest="command")

    def _add_parser(name: str, help_text: str,
                    agent_visible: bool = False,
                    user_visible: bool = True,
                    **kwargs):
        """Register a subcommand, hiding it from --help based on context.

        agent_visible=True  : shown in agent mode help (implies user_visible too)
        user_visible=False  : hidden in user mode (legacy/niche; still callable by name)
        """
        hide = (not agent_visible and AGENT_MODE) or (not user_visible and not AGENT_MODE)
        p = subparsers.add_parser(name, help=argparse.SUPPRESS if hide else help_text, **kwargs)
        if hide:
            subparsers._name_parser_map.pop(name, None)
            subparsers._choices_actions = [
                a for a in subparsers._choices_actions if a.dest != name
            ]
            subparsers._name_parser_map[name] = p  # re-add so it's still callable
        return p

    # Add command
    add_parser = _add_parser("add", "Add a new finding", agent_visible=True)
    add_parser.add_argument("content", nargs="?", help="Finding content (or use -f for file)")
    add_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    add_parser.add_argument("-p", "--project", help="Project name")
    add_parser.add_argument("-s", "--sprint", help="Sprint name")
    add_parser.add_argument("--tags", nargs="+", help="Tags")
    add_parser.add_argument("-e", "--evidence", help="Evidence/code snippet")
    add_parser.add_argument("--summary", help="One-sentence summary YOU (the author) write for "
        "this finding — preferred: you wrote the content, so summarize it in-context. "
        "If omitted, kb falls back to an extractive (no-LLM) blurb.")
    add_parser.add_argument("-f", "--file", type=Path, help="Read content from file")
    add_parser.add_argument("--no-duplicate-check", action="store_true", help="Skip duplicate checking")
    add_parser.add_argument("--no-auto-tag", action="store_true", help="Skip auto-tagging")
    add_parser.add_argument("--async", dest="async_add", action="store_true",
        help="Fire-and-forget: write to queue file and return immediately without waiting "
             "for embedding. Use when embedding server may be slow or unavailable. "
             "Note: when --file is a multi-section markdown, --async queues each section "
             "as a separate pending entry (not a single blob).")
    _split_group = add_parser.add_mutually_exclusive_group()
    _split_group.add_argument("--split", dest="split", action="store_true", default=None,
        help="Force multi-section split for markdown --file (ingest as document + sections)")
    _split_group.add_argument("--no-split", dest="no_split", action="store_true", default=False,
        help="Disable multi-section detection; always add as a single finding")

    # Search command
    search_parser = _add_parser("search", "Search findings", agent_visible=True)
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("-p", "--project", help="Filter by project")
    search_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    search_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")
    search_parser.add_argument("-l", "--long", action="store_true", help="Show full content (default: one line per result)")
    search_parser.add_argument("--exclude", nargs="*", default=[], metavar="ID",
        help="KB IDs to exclude from results (auto-loaded from session seen file)")
    search_parser.add_argument("--no-dedup", action="store_true",
        help="Disable automatic exclusion of session-seen IDs")
    search_parser.add_argument("--json", action="store_true",
        help="Output results as JSON array (full metadata, for hook/script use)")

    # List command
    list_parser = _add_parser("list", "List findings", agent_visible=True)
    list_parser.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    list_parser.add_argument("-p", "--project", help="Filter by project")
    list_parser.add_argument("-s", "--sprint", help="Filter by sprint")
    list_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    list_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")
    list_parser.add_argument("-l", "--long", action="store_true", help="Show full content (default: one line per result)")

    # Get command
    get_parser = _add_parser("get", "Get a specific finding", agent_visible=True)
    get_parser.add_argument("id", help="Finding ID")
    get_parser.add_argument("--raw", action="store_true", help="Output raw markdown")

    # Correct command
    correct_parser = _add_parser("correct", "Correct a finding (supersede)", agent_visible=True)
    correct_parser.add_argument("id", help="ID of finding to correct")
    correct_parser.add_argument("content", help="New correct content")
    correct_parser.add_argument("-e", "--evidence", help="Evidence for correction")
    correct_parser.add_argument("-r", "--reason", help="Reason for correction")

    # Delete command
    delete_parser = _add_parser("delete", "Delete a finding")
    delete_parser.add_argument("id", help="Finding ID to delete")
    delete_parser.add_argument("--force", action="store_true", help="Delete without confirmation")

    # Stats command
    _add_parser("stats", "Show database statistics")

    # Export command
    export_parser = _add_parser("export", "Export findings to JSON")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument("-p", "--project", help="Filter by project")

    # Import command
    import_parser = _add_parser("import", "Import findings from JSON")
    import_parser.add_argument("input", type=Path, help="Input file path")

    # Serve command
    serve_parser = _add_parser("serve", "Start web interface")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")


    # Ask command: LLM Q&A over KB findings
    ask_parser = _add_parser("ask", "Ask a natural language question about the KB")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("-p", "--project", help="Filter by project")
    ask_parser.add_argument("-n", "--limit", type=int, default=10, help="Max findings to consider")

    # Related command
    related_parser = _add_parser("related", "Find semantically related findings", agent_visible=True)
    related_parser.add_argument("id", help="Finding ID")
    related_parser.add_argument("-n", "--limit", type=int, default=5, help="Max results")
    related_parser.add_argument("-l", "--long", action="store_true", help="Show full content")

    # Open questions: LLM identifies research gaps from existing findings
    questions_parser = _add_parser("questions", "Identify research gaps using LLM")
    questions_parser.add_argument("query", nargs="?", help="Search query to seed findings (default: most recent)")
    questions_parser.add_argument("-p", "--project", help="Filter by project")
    questions_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of questions to generate (default: 5)")
    questions_parser.add_argument("-i", "--input", type=int, default=20, help="Number of KB entries to feed the LLM (default: 20)")

    # Review queue: surfaces untagged, stale, orphaned findings
    review_parser = _add_parser("review", "Show findings needing attention")
    review_parser.add_argument("-p", "--project", help="Filter by project")
    review_parser.add_argument("-n", "--limit", type=int, default=10, help="Max per category")


    # Refresh command: retag + resummarize + optional reembed + optional theorem backfill
    refresh_parser = _add_parser("refresh", "Retag + resummarize findings")
    refresh_parser.add_argument("targets", nargs="*", metavar="ID_OR_PROJECT",
        help="kb-ids to refresh, OR a project name (auto-detected by prefix)")
    refresh_parser.add_argument("-p", "--project", help="Restrict to one project (alias for passing project as positional arg)")
    refresh_parser.add_argument("--all", action="store_true",
        help="Regenerate ALL summaries/tags (default: only rows with NULL summary)")
    refresh_parser.add_argument("-n", "--limit", type=int, default=0,
        help="Max rows to process (0 = no limit)")
    refresh_parser.add_argument("--dry-run", action="store_true")
    refresh_parser.add_argument("--theorems", action="store_true",
        help="Also backfill statement_pure for lean theorems")
    refresh_parser.add_argument("--theorem-workers", type=int, default=8,
        help="Parallel workers for theorem backfill (default 8)")

    # Ingest command group
    ingest_parser = _add_parser("ingest", "Ingest external content into KB")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_cmd")

    ingest_lean_parser = ingest_sub.add_parser("lean",
        help="Ingest Lean theorems (proofs/ + mathlib4/ag, auto-discovered) with LLM summaries")
    ingest_lean_parser.add_argument("--dry-run", action="store_true")
    ingest_lean_parser.add_argument("--limit", type=int, default=None)
    ingest_lean_parser.add_argument("--no-summarize", action="store_true",
        help="Skip LLM summary generation (post-commit hook uses this automatically)")
    ingest_lean_parser.add_argument("--summarize-only", action="store_true",
        help="Only fill missing statement_pure for already-ingested theorems")
    ingest_lean_parser.add_argument("--files", nargs="+", metavar="FILE",
        help="Incremental: process only these absolute .lean paths")

    ingest_scripts_parser = ingest_sub.add_parser("scripts", help="Register scripts with LLM-generated purposes")
    ingest_scripts_parser.add_argument("directory", type=Path, help="Directory to scan")
    ingest_scripts_parser.add_argument("-p", "--project", default="hypercomplex")
    ingest_scripts_parser.add_argument("--dry-run", action="store_true")
    ingest_scripts_parser.add_argument("-n", "--limit", type=int, default=50)

    ingest_python_parser = ingest_sub.add_parser("python", help="Index Python symbols (.py) into the symbols table")
    ingest_python_parser.add_argument("--root", default=str(Path.cwd()),
        help="Root directory to walk for .py files (default: cwd)")
    ingest_python_parser.add_argument("--files", nargs="+", metavar="FILE",
        help="Incremental mode: process only these files")
    ingest_python_parser.add_argument("--project", default="algebraic-genesis")
    ingest_python_parser.add_argument("--dry-run", action="store_true")
    ingest_python_parser.add_argument("--with-notations", action="store_true",
        help="Also populate the physics notations table (secular-constraints only; off by default)")

    ingest_ts_parser = ingest_sub.add_parser("typescript", help="Index TypeScript/TSX symbols (.ts/.tsx) into symbols")
    ingest_ts_parser.add_argument("--root", default=str(Path.cwd()),
        help="Root directory to walk for .ts/.tsx files (default: cwd)")
    ingest_ts_parser.add_argument("--files", nargs="+", metavar="FILE",
        help="Incremental mode: process only these files")
    ingest_ts_parser.add_argument("--deleted", nargs="+", metavar="FILE",
        help="Remove all symbols rows for these deleted/renamed files")
    ingest_ts_parser.add_argument("--project", default="kb")
    ingest_ts_parser.add_argument("--dry-run", action="store_true")

    ingest_rust_parser = ingest_sub.add_parser("rust", help="Index Rust symbols (.rs) into symbols")
    ingest_rust_parser.add_argument("--root", default=str(Path.cwd()),
        help="Root directory to walk for .rs files (default: cwd)")
    ingest_rust_parser.add_argument("--files", nargs="+", metavar="FILE",
        help="Incremental mode: process only these files")
    ingest_rust_parser.add_argument("--deleted", nargs="+", metavar="FILE",
        help="Remove all symbols rows for these deleted/renamed files")
    ingest_rust_parser.add_argument("--project", default="kb")
    ingest_rust_parser.add_argument("--dry-run", action="store_true")

    ingest_tex_parser = ingest_sub.add_parser("tex", help="Index Python/Lean/Epic annotation comment blocks from TeX files")
    ingest_tex_parser.add_argument("--root", default=str(Path.home() / "Physics/claude"),
        help="Root of tex corpus (default: ~/Physics/claude)")
    ingest_tex_parser.add_argument("--files", nargs="+", metavar="FILE",
        help="Specific files to process (overrides --root glob)")
    ingest_tex_parser.add_argument("--project", default="algebraic-genesis")
    ingest_tex_parser.add_argument("--dry-run", action="store_true")

    ingest_md_parser = ingest_sub.add_parser("md", help="Ingest a markdown file into document + sections by heading tree")
    ingest_md_parser.add_argument("file", help="Markdown file to ingest (.md or .markdown)")
    ingest_md_parser.add_argument("-p", "--project", default=None, help="Project tag")
    ingest_md_parser.add_argument("--doc-type", choices=["internal", "reference", "spec", "paper", "standard"],
        default=None, help="Document type (default: inferred from front-matter or 'internal')")
    ingest_md_parser.add_argument("--title", default=None, help="Document title (default: filename stem)")
    ingest_md_parser.add_argument("--summary", default=None, help="Document summary")
    ingest_md_parser.add_argument("--dry-run", action="store_true")

    ingest_pdf_parser = ingest_sub.add_parser("pdf", help="Ingest a PDF file into document + sections (requires 'kb[pdf]' extras)")
    ingest_pdf_parser.add_argument("file", help="PDF file to ingest")
    ingest_pdf_parser.add_argument("-p", "--project", default=None, help="Project tag")
    ingest_pdf_parser.add_argument("--doc-type", choices=["internal", "reference", "spec", "paper", "standard"],
        default=None, help="Document type (default: 'reference')")
    ingest_pdf_parser.add_argument("--title", default=None, help="Document title (default: filename stem)")
    ingest_pdf_parser.add_argument("--summary", default=None, help="Document summary")
    ingest_pdf_parser.add_argument("--dry-run", action="store_true",
        help="Show outline + page count without ingesting")

    ingest_personas_parser = ingest_sub.add_parser(
        "personas",
        help="Index persona .md files (<project>/.claude/agents/personas/*.md) + staleness check",
    )
    ingest_personas_parser.add_argument(
        "roots", nargs="*", type=Path,
        help="Project root directories to scan (default: git root of cwd)",
    )
    ingest_personas_parser.add_argument("--dry-run", action="store_true",
        help="Show what would be indexed without writing")
    ingest_personas_parser.add_argument("--check", action="store_true",
        help="Report stale personas: file gone / reviewers.yaml newer / dir-count changed")
    ingest_personas_parser.add_argument("-p", "--project", dest="project_filter",
        help="Filter to a specific project name")

    # Doc command group: document navigation (list/toc/get)
    doc_parser = _add_parser("doc", "Navigate ingested documents (list, toc, get)",
                             agent_visible=True)
    doc_sub = doc_parser.add_subparsers(dest="doc_cmd")

    doc_list_parser = doc_sub.add_parser("list", help="List document roots")
    doc_list_parser.add_argument("-p", "--project", help="Filter by project")
    doc_list_parser.add_argument("--type", dest="type",
        choices=["spec", "paper", "standard", "internal", "reference"],
        help="Filter by doc type")
    doc_list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    doc_toc_parser = doc_sub.add_parser("toc", help="Print heading tree for a document")
    doc_toc_parser.add_argument("doc_id", help="Document ID")
    doc_toc_parser.add_argument("--json", action="store_true", help="Output as JSON")

    doc_get_parser = doc_sub.add_parser("get", help="Fetch a section by path")
    doc_get_parser.add_argument("doc_id", help="Document ID")
    doc_get_parser.add_argument("--path", required=True, help="Section path (e.g. '1.2.3')")
    doc_get_parser.add_argument("--subtree", action="store_true",
        help="Include all descendant sections")
    doc_get_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Bridge command group: ingest/search/promote agent bridge messages
    bridge_parser = _add_parser("bridge", "Ingest/search/promote agent bridge messages",
                                agent_visible=True)
    bridge_sub = bridge_parser.add_subparsers(dest="bridge_cmd")

    bridge_ingest_parser = bridge_sub.add_parser("ingest",
        help="Ingest ~/.agent-bridge/messages.jsonl into bridge_messages + embed substantive subset")
    bridge_ingest_parser.add_argument("--jsonl", help="Bridge messages jsonl (default: ~/.agent-bridge/messages.jsonl)")
    bridge_ingest_parser.add_argument("--since-id", type=int, default=0, metavar="N",
        help="Only process messages with id > N (incremental; default 0 = all)")
    bridge_ingest_parser.add_argument("--embed-batch", type=int, default=200, metavar="N",
        help="Embed up to N pending substantive messages this run (default 200)")

    bridge_search_parser = bridge_sub.add_parser("search",
        help="Hybrid vector+FTS search over substantive bridge messages")
    bridge_search_parser.add_argument("query", help="Search query")
    bridge_search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    bridge_search_parser.add_argument("--semantic", action="store_true",
        help="(accepted for compatibility; search is already hybrid vector+FTS)")

    bridge_promote_parser = bridge_sub.add_parser("promote",
        help="Promote a bridge message into a first-class kb finding")
    bridge_promote_parser.add_argument("id", help="bridge_messages id to promote")
    bridge_promote_parser.add_argument("-p", "--project", help="Project tag for the new finding")

    bridge_watch_parser = bridge_sub.add_parser("watch",
        help="Idle SSE bridge watcher: holds until a peer message, prints BRIDGE_WAKE, exits. "
             "Launch with run_in_background:true and NO timeout.")
    bridge_watch_parser.add_argument("agent_id", help="Your bridge agent id")

    bridge_send_parser = bridge_sub.add_parser("send",
        help="Send a bridge message via the kb-server (canonical send path).")
    bridge_send_parser.add_argument("to", help="Recipient id (or comma-list)")
    bridge_send_parser.add_argument("subject", nargs="?", default="", help="Subject")
    bridge_send_parser.add_argument("--body", default=None, help="Body (default: read stdin)")
    bridge_send_parser.add_argument("--reply", type=int, metavar="ID", help="reply_to message id")
    bridge_send_parser.add_argument("--needs-reply", action="store_true", dest="needs_reply",
        help="Mark this message as owed a reply")
    bridge_send_parser.add_argument("--from", dest="from_id", default=None,
        help="Sender id (default: AGENT_ID env / persona pin)")

    bridge_recv_parser = bridge_sub.add_parser("recv",
        help="Drain unread messages for this agent via the kb-server (usually auto-injected).")
    bridge_recv_parser.add_argument("agent_id", nargs="?", default=None,
        help="Your id (default: AGENT_ID env)")
    bridge_recv_parser.add_argument("-n", "--limit", type=int, default=50, help="Max messages")

    bridge_sub.add_parser("agents",
        help="List bridge handles with server-computed liveness (online/idle/stale/offline + last-seen).")

    bridge_announce_parser = bridge_sub.add_parser("announce", aliases=["join"],
        help="Join the bridge: kb bridge announce <id> <focus> <offering>")
    bridge_announce_parser.add_argument("id", help="your agent id")
    bridge_announce_parser.add_argument("focus", help="what you're working on now")
    bridge_announce_parser.add_argument("offering", help="what you can help peers with")

    bridge_owed_parser = bridge_sub.add_parser("owed",
        help="LIST unanswered --needs-reply messages addressed to this agent (read-only).")
    bridge_owed_parser.add_argument("agent_id", nargs="?", default=None,
        help="Your id (default: inferred from persona pin / AGENT_ID / whoami)")
    bridge_owed_parser.add_argument("--json", action="store_true", help="machine-readable output")

    bridge_clear_owed_parser = bridge_sub.add_parser("clear-owed",
        help="Clear ALL owed --needs-reply messages for this agent (stale backlog).")
    bridge_clear_owed_parser.add_argument("agent_id", nargs="?", default=None,
        help="Your id (default: inferred from persona pin / AGENT_ID / whoami)")

    # Reconcile command
    reconcile_parser = _add_parser("reconcile", "Reconcile KB with source document", user_visible=False)
    reconcile_parser.add_argument("document", type=Path, help="Source document to reconcile against")
    reconcile_parser.add_argument("-p", "--project", help="Project name")
    reconcile_parser.add_argument("--export-missing", type=Path, help="Export missing claims to file")
    reconcile_parser.add_argument("--import-missing", type=Path, help="Import missing claims from file")

    # Notation audit command
    audit_parser = _add_parser("notation-audit", "Audit notations against source document", user_visible=False)
    audit_parser.add_argument("document", type=Path, help="Source document")
    audit_parser.add_argument("-p", "--project", help="Project name")

    # lean-verify subcommand: re-run lean-audit on cited file, flag proof drift
    lean_verify_parser = _add_parser("lean-verify", "Re-run lean-audit on a lean:proven entry's cited file and flag drift")
    lean_verify_parser.add_argument("id", help="KB entry ID to verify")
    lean_verify_parser.add_argument("--search-path", nargs="*", metavar="DIR",
        help="Additional directories to search for the .lean file")

    # embed-status: show configured vs stored embedding metadata + verdict
    embed_status_parser = _add_parser(
        "embed-status",
        "Show embedding model status (configured vs stored signature)",
    )
    del embed_status_parser  # no flags needed

    # reembed: re-generate all embeddings (covers all 7 _vec tables)
    reembed_parser = _add_parser(
        "reembed",
        "Re-generate embeddings for all 7 vec tables (use after model/dim change)",
    )
    reembed_parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reembed; recreates all _vec tables if dim changed",
    )
    reembed_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in vec tables (safe only with same model)",
    )
    reembed_parser.add_argument(
        "--commit-every",
        type=int,
        default=50,
        metavar="N",
        help="Commit every N rows (default: 50)",
    )

    # Flush-pending (user/debugging only — automatic via hooks and --async spawner)
    flush_parser = _add_parser(
        "flush-pending",
        "Drain ~/.claude/pending-kb-adds/*.txt (kb-down fallback queue)",
    )
    flush_parser.add_argument(
        "--queue-dir",
        type=Path,
        default=Path.home() / ".claude" / "pending-kb-adds",
        help="Queue directory (default ~/.claude/pending-kb-adds)",
    )
    flush_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print summary line",
    )

    # configure: host-wide and per-project config UX (Phase 5, kb-2c3)
    # plan-review: content-hash verdict markers gating native plan-mode approval
    # on expert-review (epic kb-318a8b). DB-free; dispatched early like configure.
    plan_review_parser = _add_parser(
        "plan-review",
        "Plan-review verdict markers (gate native plan mode on expert-review)",
        agent_visible=True,
    )
    pr_sub = plan_review_parser.add_subparsers(dest="plan_review_cmd")
    pr_hash = pr_sub.add_parser("hash", help="Print sha256 of the normalized plan text")
    pr_hash.add_argument("plan", help="Plan file path, or - for stdin")
    pr_status = pr_sub.add_parser("status", help="Print stored verdict JSON for this plan's hash, or 'none'")
    pr_status.add_argument("plan", help="Plan file path, or - for stdin")
    pr_record = pr_sub.add_parser("record", help="Record a verdict marker for this plan's hash")
    pr_record.add_argument("plan", help="Plan file path, or - for stdin")
    pr_record.add_argument("--verdict", required=True,
        choices=["APPROVED", "APPROVED-WITH-REVISIONS", "REJECTED"])
    pr_record.add_argument("--synthesis", default="", help="One-line verdict synthesis")
    pr_record.add_argument("--blocking", nargs="*", default=[], help="Blocking issue strings")
    pr_record.add_argument("--project-root", default="", help="Project root (used by the approve-time mirror)")
    pr_record.add_argument("--epic-id", default="", help="kbt epic id this review belongs to")

    configure_parser = _add_parser(
        "configure",
        "Configure kb: embedding provider, summary mode, project setup",
    )
    # Global flags
    configure_parser.add_argument(
        "--provider",
        choices=list(["ollama-local", "voyage", "openai", "gemini", "jina", "local-llamacpp"]),
        help="Embedding provider (default: ollama-local; triggers non-interactive mode)",
    )
    configure_parser.add_argument("--model", help="Embedding model name")
    configure_parser.add_argument("--dim", type=int, help="Embedding dimension")
    configure_parser.add_argument(
        "--format", dest="format", choices=["llamacpp", "openai"],
        help="Embedding wire format (default: openai for all hosted providers)"
    )
    configure_parser.add_argument("--url", help="Embedding server URL")
    configure_parser.add_argument(
        "--llm-url",
        help="LLM completion endpoint (query-expansion + local-llm summaries); "
             "written to config.toml [llm] url. Unreachable is OK — kb degrades.",
    )
    configure_parser.add_argument(
        "--summary-mode", choices=["none", "local-llm", "subscription-sdk", "api"],
        help="Summary generation mode"
    )
    configure_parser.add_argument(
        "--key", help="Embedding API key (written to settings.local.json, gitignore-verified)"
    )
    configure_parser.add_argument(
        "--reembed", action="store_true",
        help="Run `kb reembed --force` immediately if model/dim changed"
    )
    configure_parser.add_argument(
        "--config-dir", type=Path,
        help="Global config dir (default: $CLAUDE_CONFIG_DIR or ~/.claude). Tests pass this."
    )
    # Per-project flags
    configure_parser.add_argument(
        "--project", dest="project", metavar="TAG",
        help="Per-project mode: set project tag (non-interactive, safe in background agents)"
    )
    configure_parser.add_argument(
        "--enable-tracker", action="store_true",
        help="(--project) Write .beads/config.yaml backend: kb"
    )
    configure_parser.add_argument(
        "--db-path-override", metavar="PATH",
        help="(--project) Write KB_DB=PATH to the project's .claude/settings.json env"
    )
    configure_parser.add_argument(
        "--project-dir", type=Path,
        help="(--project) Project root directory (default: cwd)"
    )
    # Server service
    configure_parser.add_argument(
        "--install-server", action="store_true",
        help="Install + enable the kb-server systemd --user service (persistent kb serve)"
    )
    configure_parser.add_argument(
        "--server-port", type=int, default=8765,
        help="(--install-server) port for the kb-server service (default: 8765)"
    )
    configure_parser.add_argument(
        "--install-wrappers", action="store_true",
        help="Install kb + kbt wrapper scripts on PATH (~/.local/bin); agents/hooks call kbt by name"
    )

    # queue-defer: set a defer_reason on a lean_work_queue row
    queue_defer_parser = _add_parser(
        "queue-defer",
        "Set or clear a defer reason on a lean_work_queue row",
        agent_visible=True,
    )
    queue_defer_parser.add_argument("row_id", nargs="?", help="lean_work_queue row id (16-char hex); not required with --list")
    queue_defer_parser.add_argument(
        "reason",
        nargs="?",
        help=(
            "Defer reason (valid: data_blocked_on:<bd-id>, design-pending:<decision>, "
            "file-conflict:<agent-id>, agent-cap, user-gate:<adj>, verify-first:<row-id>). "
            "Omit to CLEAR the defer (re-activates the row)."
        ),
    )
    queue_defer_parser.add_argument(
        "detail",
        nargs="?",
        help="Optional free-text detail appended after reason",
    )
    queue_defer_parser.add_argument(
        "--list", action="store_true",
        help="List all deferred rows (read-only)",
    )

    # surface: unified multi-source semantic surfacing (code symbols + findings + bridge)
    surface_parser = _add_parser(
        "surface",
        "Unified semantic surface: code symbols + findings + bridge memory",
        agent_visible=True,
    )
    # Producer modes (kb-xob.1) — mutually exclusive with legacy --query positional
    surface_parser.add_argument("query", nargs="?", default=None, help="Search query (legacy --query mode)")
    surface_parser.add_argument("--prompt", metavar="TEXT",
        help="What kb-prompt-surface would inject for this user prompt (sim>=0.42, top-3)")
    surface_parser.add_argument("--analysis", metavar="TEXT",
        help="What kb-analysis-surface would inject for this assistant text (INTENT_RX + sim>=0.62)")
    surface_parser.add_argument("--file", metavar="PATH",
        help="What symbol_surface would inject after Read of this file (RETIRED/NOTATION)")
    surface_parser.add_argument("--issues", metavar="TEXT",
        help="What open_issues_surface would inject for this dispatch/bridge prompt text")
    surface_parser.add_argument("--bridge", metavar="ID_OR_TEXT",
        help="What bridge-inject would surface for a bridge message id (int) or raw text")
    surface_parser.add_argument("--all", metavar="TEXT", dest="all_input",
        help="Run ALL producers against this input, each labeled; lists producers that did NOT fire")
    surface_parser.add_argument("-n", "--limit", type=int, default=8,
        help="Max results per source (default: 8)")
    surface_parser.add_argument("-p", "--project", help="Filter findings + symbols + issues by project")
    surface_parser.add_argument("--sources", default="code,findings,bridge",
        metavar="SOURCES",
        help="Comma-separated sources for --query mode: code,findings,bridge (default: all three)")
    surface_parser.add_argument("--min-sim", type=float, default=0.45, dest="min_sim",
        help="Minimum cosine similarity floor for --query mode (default: 0.45)")
    surface_parser.add_argument("--json", action="store_true",
        help="Output as structured JSON for hook/script consumption")

    args = parser.parse_args()

    if args.help or not args.command:
        _print_main_help()
        sys.exit(0 if args.help else 1)

    # Async `kb add`: resolve content early (needed for both sync and async paths).
    # Lean tag validation runs here too — before any I/O — for both sync and async.
    _add_content: str = ""
    if args.command == "add":
        if args.file:
            _add_content = args.file.read_text().strip()
        elif args.content:
            _add_content = args.content
        else:
            print("Error: Either content or --file required")
            sys.exit(1)
        if not _add_content:
            print("Error: empty content")
            sys.exit(1)

        _lean_errors = _cmd_lean.validate_lean_tags(args.tags, _add_content, args.evidence)
        if _lean_errors:
            for _err in _lean_errors:
                print(f"Error [lean: tag]: {_err}", file=sys.stderr)
            sys.exit(1)

        # Multi-section markdown detection for `kb add -f <file.md>`.
        # If the file is markdown with >=2 heading-bounded sections and --no-split
        # is not set, route to the markdown chunker (document + sections) instead
        # of the single-finding path.
        _do_split = False
        if args.file and not getattr(args, "no_split", False):
            _md_exts = {".md", ".markdown"}
            if args.file.suffix.lower() in _md_exts:
                from kb.ingest.markdown import count_heading_sections
                _n_sections = count_heading_sections(_add_content)
                _do_split = _n_sections >= 2 or getattr(args, "split", False)

        if _do_split:
            from kb.ingest.markdown import ingest_markdown_file
            if args.async_add:
                # Queue each section as its own pending entry using _queue_async_add.
                # We need the raw intermediate list to do this without hitting the DB.
                from kb.ingest.markdown import (
                    _parse_front_matter, _parse_sections, _build_intermediate,
                    _compute_paths,
                )
                _meta, _body = _parse_front_matter(_add_content)
                _raw_secs = _parse_sections(_body)
                _inter = _build_intermediate(_raw_secs)
                _compute_paths(_raw_secs, _inter)
                for _entry in _inter:
                    _queue_async_add(
                        content=_entry["content"] or "",
                        finding_type=args.type,
                        project=args.project,
                        sprint=args.sprint,
                        tags=args.tags,
                        evidence=args.evidence,
                        summary=None,
                    )
                print(f"Queued: {len(_inter)} sections from {args.file.name}")
                sys.exit(0)
            else:
                _doc_id, _sec_ids = ingest_markdown_file(
                    args.file,
                    db_path=args.db,
                    project=args.project,
                )
                print(f"doc-id: {_doc_id}")
                print(f"sections: {len(_sec_ids)}")
                for _sid in _sec_ids:
                    print(f"  {_sid}")
                sys.exit(0)

        if args.async_add:
            _queue_async_add(
                content=_add_content,
                finding_type=args.type,
                project=args.project,
                sprint=args.sprint,
                tags=args.tags,
                evidence=args.evidence,
            )
            sys.exit(0)

    # plan-review: pure filesystem marker store — no KnowledgeBase/embedding.
    # Dispatch early so the PreToolUse plan gate stays cheap (epic kb-318a8b).
    if args.command == "plan-review":
        from kb.cli.commands import plan_review as _cmd_plan_review
        sys.exit(_cmd_plan_review.run_plan_review(args))

    # configure: does not need KnowledgeBase — dispatch early
    if args.command == "configure":
        from kb.configure import configure_main
        sys.exit(configure_main(args))

    # Initialize KB
    kb = KnowledgeBase(
        db_path=args.db,
    )

    # Dispatch table: command name -> callable(kb, args)
    # Commands that need extra context (closures / helpers) are wrapped as lambdas.
    _dispatch = {
        "add": lambda kb, args: _cmd_findings.run_add(
            kb, args, _add_content, _queue_async_add),
        "search": lambda kb, args: _cmd_findings.run_search(
            kb, args, _load_session_seen_ids, format_results, format_finding),
        "list": lambda kb, args: _cmd_findings.run_list(
            kb, args, format_results, format_finding),
        "get": _cmd_findings.run_get,
        "correct": _cmd_findings.run_correct,
        "delete": _cmd_findings.run_delete,
        "stats": _cmd_admin.run_stats,
        "export": _cmd_admin.run_export,
        "import": _cmd_admin.run_import,
        "embed-status": _cmd_admin.run_embed_status,
        "reembed": _cmd_admin.run_reembed,
        "flush-pending": _cmd_admin.run_flush_pending,
        "review": _cmd_maintenance.run_review,
        "refresh": lambda kb, args: _cmd_maintenance.run_refresh(
            kb, args, _fetch_refresh_rows, _run_refresh, _backfill_statement_pure),
        "ask": _cmd_maintenance.run_ask,
        "questions": _cmd_maintenance.run_questions,
        "related": lambda kb, args: _cmd_maintenance.run_related(
            kb, args, _fmt_one_line, format_finding),
        "ingest": lambda kb, args: _cmd_ingest.run_ingest(kb, args, ingest_parser),
        "doc": lambda kb, args: _cmd_doc.run_doc(kb, args, doc_parser),
        "bridge": lambda kb, args: _cmd_bridge.run_bridge(kb, args, bridge_parser),
        "reconcile": _cmd_misc.run_reconcile,
        "notation-audit": _cmd_misc.run_notation_audit,
        "lean-verify": _cmd_lean.run_lean_verify,
        "queue-defer": _cmd_lean.run_queue_defer,
        "serve": _cmd_serve.run_serve,
        "surface": _cmd_surface.run_surface,
    }

    handler = _dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

    try:
        handler(kb, args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
