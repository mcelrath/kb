#!/usr/bin/env python3
"""kb-2os.6 (P2b): copy the 3 tracker/lifecycle hooks into the plugin, converting
bd->kbt. kbt is bd-compatible (passes through to dolt for non-migrated projects,
kb-native for migrated), so the conversion is safe either way. Precise per-file
replacements — NO blind sed (the id-regex in block-followup matches kbt ids
already via its project-prefix branch and must NOT be touched)."""
import pathlib

SRC = pathlib.Path("/home/mcelrath/Projects/ai/claude/hooks")
DST = pathlib.Path("/home/mcelrath/Projects/ai/kb/hooks/scripts")

# (src_rel, dst_name, [(old, new), ...])
JOBS = [
    ("git/bd-lifecycle.sh", "kbt-lifecycle.sh", [
        ("auto-closes bd issues referenced in git commit messages",
         "auto-closes kbt issues referenced in git commit messages"),
        ('bd close "$id"', 'kbt close "$id"'),
        ("Auto-closed bd issue:", "Auto-closed kbt issue:"),
    ]),
    ("session/session-followups.sh", "session-followups.sh", [
        ("require those to be real bd issues", "require those to be real kbt issues"),
        ("command -v bd >/dev/null 2>&1 || exit 0", "command -v kbt >/dev/null 2>&1 || exit 0"),
        ("bd list --limit=1 >/dev/null 2>&1 || exit 0", "kbt list --limit=1 >/dev/null 2>&1 || exit 0"),
        ("bd list --type=epic --status=closed --json", "kbt list --type=epic --status=closed --json"),
        ('bd dep list "$epic_id" --json', 'kbt dep list "$epic_id" --json'),
        ("# bd dep list <epic> shows", "# kbt dep list <epic> shows"),
        ("Run: bd ready  /  bd show <id>", "Run: kbt ready  /  kbt show <id>"),
    ]),
    ("guards/block-followup-without-bd-id.sh", "block-followup-without-bd-id.sh", [
        # gate logic + id-regex UNCHANGED (matches kbt ids); convert terminology only.
        ("Blocks plan writes that defer work without a bd-ID.",
         "Blocks plan writes that defer work without a tracker-ID (kbt)."),
        ("requires a bd-ID", "requires a kbt-ID"),
        ("If any reference lacks a\n# nearby bd-ID", "If any reference lacks a\n# nearby kbt-ID"),
        ("requires those to be real bd issues", "requires those to be real kbt issues"),
        ("bd-IDs make the work", "kbt-IDs make the work"),
        ("'bd ready' surfaces it", "'kbt ready' surfaces it"),
        ("without a bd-ID anchor.", "without a kbt-ID anchor."),
        ("must be a real bd issue", "must be a real kbt issue"),
        ("Plans refer to follow-ups by bd-ID.", "Plans refer to follow-ups by kbt-ID."),
        ("       bd create --title=", "       kbt create --title="),
        ("Create the bd issues.", "Create the kbt issues."),
    ]),
]

for src_rel, dst_name, repls in JOBS:
    text = (SRC / src_rel).read_text()
    for old, new in repls:
        if old not in text:
            print(f"  WARN {dst_name}: pattern not found: {old[:50]!r}")
            continue
        text = text.replace(old, new)
    (DST / dst_name).write_text(text)
    print(f"{dst_name}: written (from {src_rel})")
