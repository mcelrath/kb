"""CLI handlers for admin commands: stats, export, import, embed-status, reembed, flush-pending."""

import os
import sys
from pathlib import Path

import kb.cli.output as output


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def run_stats(kb, args) -> None:
    stats = kb.stats()
    print(f"{output.c('Database:', 'bold')} {stats['db_path']}")
    print(f"{output.c('Total findings:', 'bold')} {output.c(str(stats['total']), 'cyan')}")
    print(f"  {output.c('Current:', 'dim')}    {stats['current']}")
    print(f"  {output.c('Superseded:', 'dim')} {stats['superseded']}")
    no_sum = stats.get('no_summary', 0)
    no_emb = stats.get('no_embedding', 0)
    warn_sum = output.c(str(no_sum), 'yellow') if no_sum else str(no_sum)
    print(f"  {output.c('No summary:', 'dim')} {warn_sum}"
          + (f"  {output.c('(run: kb refresh -p PROJECT)', 'yellow')}" if no_sum else ""))
    if stats.get('no_summary_by_project'):
        for proj, cnt in stats['no_summary_by_project'].items():
            print(f"    {output.c(proj, 'dim')}: {cnt}")
    warn_emb = output.c(str(no_emb), 'yellow') if no_emb else str(no_emb)
    print(f"  {output.c('No embed:', 'dim')}   {warn_emb}"
          + (f"  {output.c('(run: kb refresh --all -p PROJECT)', 'yellow')}" if no_emb else ""))
    if stats.get('no_embedding_by_project'):
        for proj, cnt in stats['no_embedding_by_project'].items():
            print(f"    {output.c(proj, 'dim')}: {cnt}")
    print(f"\n{output.c('By type:', 'bold')}")
    for t, count in sorted(stats['by_type'].items()):
        print(f"  {output.c(t, 'cyan')}: {count}")
    print(f"\n{output.c('By project:', 'bold')}")
    for p, count in sorted(stats['by_project'].items()):
        print(f"  {output.c(p, 'cyan')}: {count}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def run_export(kb, args) -> None:
    result = kb.export_findings(args.output, project=args.project)
    print(f"Exported {result['count']} findings to {args.output}")


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def run_import(kb, args) -> None:
    result = kb.import_findings(args.input)
    print(f"Imported {result['imported']} findings ({result['skipped']} skipped as duplicates)")


# ---------------------------------------------------------------------------
# embed-status
# ---------------------------------------------------------------------------

def run_embed_status(kb, args) -> None:
    status = kb.embedding_status()
    verdict = status["verdict"]
    configured = status["configured"]
    stored = status.get("stored")

    print(f"Configured: format={configured['format']} url={configured['url']} "
          f"model={configured['model'] or '(none)'} dim={configured['dim']}")
    if stored:
        print(f"Stored:     format={stored['format']} url={stored['url']} "
              f"model={stored['model'] or '(none)'} dim={stored['dim']} "
              f"updated={stored.get('updated_at', '?')}")
    else:
        print("Stored:     (no embedding_meta row)")

    print(f"Verdict:    {verdict}")
    print(f"Message:    {status['message']}")

    if verdict in ("mismatch-same-dim", "mismatch-dim-change", "no-meta"):
        print("\nRun: kb reembed --force", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# reembed
# ---------------------------------------------------------------------------

def run_reembed(kb, args) -> None:
    status = kb.embedding_status()
    verdict = status["verdict"]
    if verdict == "ok" and not args.force:
        print(f"Embedding metadata OK (no change detected). Use --force to reembed anyway.")
        sys.exit(0)

    result = kb.reembed_all(
        resume=args.resume,
        commit_every=args.commit_every,
    )
    total_updated = sum(v.get("updated", 0) for v in result.values())
    total_failed = sum(v.get("failed", 0) for v in result.values())
    print(f"reembed: {total_updated} re-embedded, {total_failed} failed")
    for table, s in result.items():
        print(f"  {table}: updated={s['updated']} failed={s['failed']} total={s['total']}")


# ---------------------------------------------------------------------------
# flush-pending
# ---------------------------------------------------------------------------

def _discover_kb_lane_files() -> list[Path]:
    """Discover per-project .kb/*.md files to drain via flush-pending.

    Discovery rule: scan every git-tracked project root that contains a `.kb/`
    subdirectory.  Project roots are located by walking the parent directories of
    the queue dir's siblings AND a small hard-coded search base list
    (~/.claude and ~/Projects) so that no external config file is needed.

    Concrete steps:
      1. Walk `~/Projects` and `~/.claude` up to depth 3 looking for directories
         that contain both `.git` (or `.kbt`) and `.kb/`.
      2. Collect all `<root>/.kb/*.md` files (non-recursive; only immediate children).
      3. Skip files named `.flush.lock` or starting with `.`.

    On-disk contract: the agent Writes `<project>/.kb/<slug>.md`; flush-pending
    ingests it via `ingest_markdown_file` (doc_type='internal') and removes it on
    success.  On failure the file is left in place for the next run (mirrors the
    `.txt` no-delete-on-failure semantics).
    """
    search_bases = [
        Path.home() / "Projects",
        Path.home() / ".claude",
    ]
    results: list[Path] = []
    seen: set[Path] = set()
    # Glob ONLY for `.kb` directories at depths 0-3 (directory-name patterns
    # expand level-by-level — no full recursive file walk; rglob("*") over
    # ~/Projects would enumerate every file in llama.cpp/mathlib4/... on each run).
    patterns = (".kb", "*/.kb", "*/*/.kb", "*/*/*/.kb")
    for base in search_bases:
        if not base.is_dir():
            continue
        for pat in patterns:
            for kb_dir in base.glob(pat):
                if not kb_dir.is_dir():
                    continue
                cand = kb_dir.parent
                if not ((cand / ".git").exists() or (cand / ".kbt").is_dir()):
                    continue
                for md in sorted(kb_dir.glob("*.md")):
                    if md.name.startswith("."):
                        continue
                    rp = md.resolve()
                    if rp in seen:  # ~/.claude is a symlink into ~/Projects — dedup
                        continue
                    seen.add(rp)
                    results.append(md)
    return results


def run_flush_pending(kb, args) -> None:
    """Drain the pending-adds queue AND per-project .kb/*.md agent-report files.

    Sources:
      1. Global txt queue (``args.queue_dir/*.txt``): header (``# k: v``) +
         blank line + body → one ``kb.add`` call per file.  On success the file
         is deleted; on failure it is renamed back to ``.txt`` (no-delete-on-
         failure).  Protected by an flock so concurrent runs are safe.

      2. Per-project ``.kb/*.md`` lane: each ``<project-root>/.kb/<slug>.md``
         file is ingested as a ``doc_type='internal'`` document via
         ``kb.ingest.markdown.ingest_markdown_file``.  Discovery is performed by
         ``_discover_kb_lane_files()``.  On success the file is removed; on
         failure it is left in place.  The same health-gate and flock as the txt
         source apply (both sources run inside the same lock).
    """
    import fcntl
    from urllib.parse import urlsplit, urlunsplit
    from urllib.request import urlopen

    qdir = args.queue_dir
    if not qdir.is_dir():
        if not args.quiet:
            print(f"queue dir empty/missing: {qdir}")
        sys.exit(0)

    lock_path = qdir / ".flush.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.quiet:
            print("another flush-pending is running; exiting")
        os.close(lock_fd)
        sys.exit(0)

    os.environ.setdefault("KB_EMBED_TIMEOUT", "900")

    # --- Source 1: global *.txt queue ---
    txt_files = sorted(qdir.glob("*.txt"))

    # --- Source 2: per-project .kb/*.md files ---
    kb_md_files = _discover_kb_lane_files()

    if not txt_files and not kb_md_files:
        if not args.quiet:
            print("no pending entries")
        sys.exit(0)

    parts = urlsplit(kb.embedding_url)
    health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
    try:
        with urlopen(health_url, timeout=5) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"health HTTP {resp.status}")
    except Exception as e:
        total_pending = len(txt_files) + len(kb_md_files)
        if not args.quiet:
            print(f"embedding server not healthy ({health_url}): {e}; leaving {total_pending} file(s) queued")
        sys.exit(0)

    ok = fail = 0

    # --- Drain txt queue ---
    for f in txt_files:
        claimed = f.with_suffix(f.suffix + ".flushing")
        try:
            f.rename(claimed)
        except OSError:
            continue
        try:
            raw = claimed.read_text()
            headers = {}
            body_lines = []
            in_body = False
            for line in raw.splitlines():
                if in_body:
                    body_lines.append(line)
                    continue
                if line.startswith("# ") and ":" in line:
                    k, v = line[2:].split(":", 1)
                    headers[k.strip().lower()] = v.strip()
                elif line.strip() == "":
                    in_body = True
                else:
                    body_lines.append(line)
                    in_body = True
            content = "\n".join(body_lines).strip()
            if not content:
                raise ValueError("empty content")
            tags_str = headers.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
            result = kb.add(
                content=content,
                finding_type=headers.get("type", "discovery"),
                project=headers.get("project") or None,
                sprint=headers.get("sprint") or None,
                tags=tags,
                evidence=headers.get("evidence") or None,
                summary=headers.get("summary") or None,
                check_duplicate=False,
            )
            finding_id = result.get("id") if isinstance(result, dict) else result
            if not finding_id:
                raise RuntimeError(f"kb.add returned no id: {result}")
            claimed.unlink()
            ok += 1
            if not args.quiet:
                print(f"flushed {f.name} -> {finding_id}")
        except Exception as e:
            try:
                claimed.rename(f)
            except OSError:
                pass
            fail += 1
            if not args.quiet:
                print(f"FAILED {f.name}: {e}")

    # --- Drain .kb/*.md lane ---
    from kb.ingest.markdown import ingest_markdown_file
    from kb.config import load_config

    cfg = load_config()
    db_path = cfg.db_path

    for md_file in kb_md_files:
        try:
            doc_id, section_ids = ingest_markdown_file(
                md_file,
                db_path=db_path,
                doc_type="internal",
            )
            md_file.unlink()
            ok += 1
            if not args.quiet:
                print(f"flushed .kb/{md_file.name} -> {doc_id} ({len(section_ids)} sections)")
        except Exception as e:
            # No-delete-on-failure: leave the file in place for next run.
            fail += 1
            if not args.quiet:
                print(f"FAILED .kb/{md_file.name}: {e}")

    total = len(txt_files) + len(kb_md_files)
    print(f"flush-pending: {ok} ok, {fail} failed, {total} total")
    sys.exit(0 if fail == 0 else 1)
